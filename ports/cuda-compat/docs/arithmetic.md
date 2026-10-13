# browser-compat-v1 arithmetic

This contract identifies the executed behavior of a pinned browser reference
on the validation RTX 2060. It is distinct from the canonical upstream numerical
contract. The complete-network compatibility claim applies only to independently
identified captures that pass the strict parity checks.

| Operation | Supported behavior |
|---|---|
| Model weights | Decode packed E4M3 to FP16 once; retain signed zero and the source model's specified NaN-code replacement. |
| F13 GEMMs | Align/truncate products in K16 groups; retain FP32 between groups and round final raw-half publication toward zero. |
| Residuals | Seed with the FP32 residual-times-scale value. |
| Partitioned GEMMs | Reset each partition, seed residual only in the first, and sum FP32 partials in order. |
| E4 publication | Quantize directly from the appropriate FP32 accumulator using E4 round-to-nearest-even semantics. |
| SiLU | Use the observed FP32 fused inner/polynomial operations, final FP32 multiply, RTZ half result and direct E4 publication. Preserve negative-zero behavior. |
| Window attention | Keep real half arithmetic RNE; the observed FP32 pair-square sum, inverse norm, softmax reciprocal and learned-scale conversions use half RTZ. |
| Global attention | Use the canonical manual-RNE/F13 path. |
| FP16 adapter and head | Software F24 K8 product alignment and publication. |

Inference combines FP16 weight storage with software E4M3 publication and
F13/F24 reductions. These reproduce the reference arithmetic on SM75 without
native FP8 conversion or matrix instructions. The API identifies this contract
as `browser-compat-v1` (policy 6).

The reference browser capture enabled `vulkan_enable_f16_on_nvidia`. Some
intermediate half casts in generated source did not describe the observed
executed behavior. The reference identity therefore includes browser, driver,
source, model, geometry and input hashes. A changed browser or driver requires
newly identified captures and a review of compatibility; it cannot silently
replace expected values.

## Three different validation claims

1. **Canonical operator arithmetic:** the public upstream `numerics.bin` contains
   scalar cases, 16,384 FDP8 K16 answers and 16,384 FD16 K8 answers. Its SHA256 is
   `f82c0df26adc5a3b2222b632579314afd0c36cf7b142782a2bd5dcb746eb4b0b`.
2. **Browser network compatibility:** all required captured boundaries and the
   full padded four-channel head must match the pinned browser reference.
3. **Original/Vulkan network parity:** requires its own independent capture pack.
   Passing the first two checks does not establish this third claim.

Canonical and browser witnesses remain separately named, including cases that
distinguish their accumulation/publication behavior. Expected values must come
from the independent reference or a specified independent arithmetic oracle.
The candidate's output must never be used to regenerate its own golden answer.

An E4 byte-exact comparison checks original reference codes against independently
encoded, exactly representable candidate values. Decoded equality is a separate
claim because a NaN code can collapse to a decoder's zero substitute. Signed
zeros remain significant throughout.
