# Frozen Marlin module

The bundled `marlin-sm80.cubin` is the vLLM 0.28.0 Marlin image already validated by Imparo on SM86. Original source revision: `2cf0a6915ce544dc493a0990f2ea38d81601128a` ([Marlin source](https://github.com/vllm-project/vllm/tree/2cf0a6915ce544dc493a0990f2ea38d81601128a/csrc/libtorch_stable/quantization/marlin)). It was extracted without arithmetic changes from the installed `_C_stable_libtorch.abi3.so` image number 51 for SM80. Only the fixed function named in `marlin_frozen.cuh` is selected. Redistribution license: [Apache-2.0](LICENSE.marlin).

Image size: 4,348,832 bytes. SHA-256: `9521406b46b918c23450e6199ed973149c88fcb2e33e5a7d8348212f7694cf37`.

The Windows speculative build embeds these bytes. The existing numerical-family knob is off by default; bundling the image does not admit a performance or quality profile. Rust authenticates the same byte slice passed synchronously to the existing owner via CUDA Driver `cuModuleLoadDataEx`; native publishes its loaded identity only after resolving the function and its resources. The old lab file override must contain this identical image and cannot authorize a production correctness receipt.
