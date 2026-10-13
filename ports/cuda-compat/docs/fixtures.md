# Parity fixtures

The `parity` command compares the full padded four-channel head and all 75 E4
boundaries against an independent capture. It runs production graph replays and
an instrumented context; every head and boundary must match exactly.

## Full-network capture format

Pass a `fixture.json` file or its directory. Available captures use these exact
fields; hashes are lowercase SHA256 strings and commits are lowercase 40-digit
Git IDs.

| Field | Required value or contents |
|---|---|
| `format` | `opendlss-cuda-feature-fixture-v1` |
| `availability` | `available` |
| `checks` | `["head", "all-75-boundaries"]` |
| `reference` | Reference identity described below |
| `model` | `source_manifest_sha256`, `canonical_manifest_sha256`, `canonical_blob_sha256` |
| `geometry` | `mode`, `valid_width`, `valid_height`, `full_width`, `full_height` |
| `features` | Tensor descriptor for float32 features `[full_height, full_width, 16]` |
| `head` | Tensor descriptor for float32 head `[full_height, full_width, 4]` |
| `boundaries` | Mapping of every required boundary name to its tensor descriptor |

`reference` contains `domain`, `upstream_commit`, `capture_manifest_sha256`,
`producer`, `producer_sha256`, `device`, and `driver`. The `browser-compat-v1`
domain also requires `browser_version`, `browser_binary_sha256`, and
`shader_source_sha256`; omit those three fields for `original-vulkan` captures.
Expected outputs must come from that reference, not the candidate backend.

`geometry.mode` is `upstream` or `legacy_belvedere`. The valid and padded
sizes must match that rule. `required_boundary_shapes(geometry)` in
`opendlss_cuda.fixture` supplies the required shapes. Names are `block-0` through
`block-69`, plus `transition-0-1`, `transition-4-5`, `transition-8-9`,
`transition-14-15`, and `transition-22-23`; block 70 is the full head.

Each descriptor contains exactly:

```json
{
  "file": "features.npy",
  "file_sha256": "<SHA256 of the file>",
  "bytes": 6553728,
  "array_sha256": "<SHA256 of contiguous array bytes>",
  "format": "f32-npy",
  "shape": [320, 320, 16]
}
```

Use the actual byte length and shape. Features and heads are C-order,
little-endian float32 NPY v1/v2. Boundaries use `format: "e4"`, raw uint8 files
and `[rows, channels]` shapes; their file and array hashes are identical. Each
descriptor names a distinct file relative to the capture directory.

The loader checks shapes, hashes and finite data before opening CUDA. It rejects
child symlinks, path traversal, duplicate/unknown JSON fields and trailing array
payloads. Files are limited to 128 MiB each and 512 MiB per pack. Candidate
boundary halves must be exactly E4-representable; comparison checks the original
reference bytes, retaining signed-zero and NaN-code distinctions.

Exit codes are 0 for exact agreement, 1 for mismatch, 2 for invalid input or
runtime failure, and 3 for an unavailable reference. An unavailable capture is:

```json
{"format":"opendlss-cuda-feature-fixture-v1","availability":"unavailable",
 "reference_domain":"original-vulkan","reason":"Capture not supplied"}
```

No model weights, images or full-network captures are bundled. GPU validation
uses an externally supplied browser capture; original/Vulkan parity remains
unverified.

## Public operator fixtures

- `numerics.bin` is the upstream scalar/F13/F24 reference. `upstream_numeric.h`
  provides its independent CPU oracle; `LICENSE.upstream` preserves its MIT notice.
- `browser-silu-observed.bin` records output-half bits, E4 code and a reserved
  zero byte for each half-bit input. Its JSON sidecar identifies the browser
  capture and hashes. Supported cases are finite-half inputs.
- `f13/` contains independent synthetic group and GEMM witnesses for residuals,
  partitions, tails and long reductions. `report.json` records hashes/layouts.

`tools/generate_numeric_fixtures.py --verify tests/fixtures` checks the fixed
references and regenerates the synthetic files byte for byte. Its oracle uses
exact dyadic products and integer-grid truncation, independently of CUDA helpers.
The native tests also include literal rounding and cancellation witnesses.
