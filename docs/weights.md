# Weights: the model directory and the layouts

`src/nr_model.cpp` reads the model directory and hands the kernels plain matrices. Everything here is about the
gap between "how the weights are stored" and "what a cooperative-matrix load wants".

## The directory

`manifest.json` holds:

* `totals.blockCount`: 71 (31 for the small network of build 310.9.16);
* `stages[]`: eleven files of packed E4M3 bytes (`id`, `file`, `packedByteLength`, `sha256`), verified on load
  unless `--no-verify`;
* `tensors[]`: 153 records (77 in the small network), `blockN.layerM.parameter`, each a `(stage, stageOffset, byteLength)` slice.

One record holds a whole layer: its weight matrices, its per-channel f16 scale vectors, its attention prior and
its per-head f32 scales, laid end to end. `FusedLayout` in `nr_graph.cpp` is that layout; every offset in it is
a sum of the shapes before it, and the sum reproduces every tensor's `byteLength` exactly, which is the check
that the layout is right.

### A window block of C channels (`fusedLayout`)

| offset | bytes | contents |
| --- | --- | --- |
| `expand` = 0 | `E·C·128` (experts) or `C·128` | FFN W1 |
| `contractWeights` | `E·128·32` + `C·C` (experts) or `128·C` | FFN W2 (+ W3) |
| | 16 | zero padding |
| `ffnCosSkip` | `2C` | FFN skip scale, f16 per channel |
| | 16 | zero padding |
| `qkv` | `3C²` | fused Q\|K\|V projection, head-major |
| `relative` | `heads · 8192` | the 64x64 f16 attention prior per head |
| `scale` | `alignUp(4·heads, 16)` | the learned f32 attention scale per head |
| `projection` | `C²` | attention output projection |
| `attnCosSkip` | `2C` | attention skip scale, f16 per channel |
| | 16, or the `C -> 2C` transition matrix | trailing padding, or the next stage's input projection |

`E = C / 32` experts for `C >= 64`; the 32-channel blocks are dense and have no W3. The variants:

* **block 0** inserts the `16 -> 32` f16 input adapter after W2 (`preFusedLayout`);
* **the first block of a decoder stage** inserts the `2C -> C` upsample matrix and a second scale vector for the
  skip merge (`upsampleFusedLayout`);
* **block 70** inserts the two blend scale vectors and appends the `32 -> 4` f16 head (`postFusedLayout`);
* **the 512 blocks** split over four records: `layer0` the eight branch MLPs, `layer1` the contraction plus its
  skip scale, `layer2` qkv + priors + head scales, `layer3` the projection plus its skip scale;
* **the ViT blocks** use `layer0` expand, `layer1` contract (+ skip scale), `layer2` (head scales **first**, then
  qkv), `layer4` projection (+ skip scale). `layer3` is two unread bytes;
* **block 30 `layer4`** is the `512 -> 1024` matrix into the ViT, and **block 39 `layer0`** the `1024 -> 512`
  matrix out of it plus the skip scale.

## The FP8 weight packing

Weights are stored as MMA fragments, not as matrices. `packedWeightIndex(k, n, N)`:

```
byte = kTile·(N·32)                      # kTile = k / 32
     + nTile·4096 + nHalf·2048 + nGroup·512     # nTile = n / 128, nHalf = (n%128)/64, nGroup = (n%64)/16
     + lane·16 + byteInLane
lane       = (n % 8)·4 + (k % 16)/4
byteInLane = ((n % 16)/8)·8 + ((k % 32)/16)·4 + (k % 4)
```

The innermost 512 bytes are one `m16n8k32` **B fragment pair** (16 columns x 32 k over 32 lanes, 16 bytes each:
two 8-column halves x two 16-k halves x 4 bytes). The host walks this and writes a plain matrix instead:
`[K/batchK][batchK/32][N][32]` (k32-tile-major, so a B tile is one contiguous 16-byte-aligned block) or
`[K/batchK][N][batchK]` (N-major) when a kernel wants single multiplies.

## The chained activation index

Every FP8 GEMM's A operand is indexed by a permutation within each group of 32 channels:

```
nativeChainedInputIndex: within a 32-group, rotate bits 1..3 of the index:  (b3, b1, b2) <- (b1, b2, b3)
  0 1 2 3 4 5 6 7 8 9 10 ... -> 0 1 8 9 2 3 10 11 4 5 12 ...
```

Rather than permuting activations at runtime, the host applies the **inverse** to the K rows of every weight
matrix, so the kernels load A in natural channel order. This is legitimate because the permutation stays inside
each group of 16 products, which is the grouping the F13 arithmetic sums over
([numerics.md](numerics.md#tensor-core-arithmetic)). Reordering inside a group would change the result.
`Model`'s constructor asserts the permutation is invertible.

## The f16 matrices

The `16 -> 32` input adapter and the `32 -> 4` head are stored as `m16n8k16` f16 fragments, tiled 16x16 in
`(k, n)` row-major order:

```
half = tile·256 + lane·8 + ((n % 16)/8)·4 + fragment,   tile = (k/16)·ceil(N/16) + n/16
lane = (n % 8)·4 + (k % 8)/2,   fragment = (k % 16 >= 8 ? 2 : 0) + (k % 2)
```

The two matrices are the two ways this degenerates (one k tile by two n tiles; two k tiles by one n tile), which
is why `packedF16WeightIndex` is one function and not two layouts.

## The attention prior

Stored as 16x16 fragment tiles with **both** token indices in the physical (4x4-tiled) order.
`Model::relativeBias` rewrites it to `[head][64 natural query][64 physical key]` f16, because that is what the
kernels index: queries are the A rows of `S = Q Kᵀ` and arrive in natural window order, keys are the B columns
and are staged in physical order (which is also the order the softmax reduces in).

```
tiledToken(t): x = t % 8, y = t / 8  ->  (y/4)·32 + (x/4)·16 + (y%4)·4 + (x%4)
```

## Caching and lifetime

Every re-layout is done once on the host at load time and uploaded to a device-local buffer, keyed by
`(tensor, offset, shape, batchK, layout)` in `Model::matrices_`. Re-recording the graph re-requests the same
keys and gets the same buffers. The `Model` owns them and destroys them in its destructor; nothing in the graph
outlives it. Loading the whole model costs a few seconds, dominated by the SHA-256 verification.
