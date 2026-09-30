# Frozen Gemma assistant cluster assets

Authors: Google DeepMind. Source model: `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`, revision `65892304d4eb7762acc45257a327885f7535e584`.

[Fixed upstream model card](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant/blob/65892304d4eb7762acc45257a327885f7535e584/README.md) declares `license: apache-2.0` and links [the official license](https://ai.google.dev/gemma/docs/gemma_4_license), which redirects to [Apache 2.0 terms](https://ai.google.dev/gemma/apache_2). The exact model-card bytes retrieved for this delivery have SHA256 `32793987752694ea9de0d592b6e1a4903d41eb4dea53d0472c30a53542904c86`. The Apache-2.0 text is retained in [LICENSE](LICENSE).

Only these two tensors are redistributed. They preserve the fixed source values and canonical token ordering; they are not trained, re-quantized or altered for benchmark inputs.

| Source tensor | Original representation and SHA256 | Bundled representation and SHA256 |
|---|---|---|
| `masked_embedding.centroids.weight` | BF16, shape 2048 x 256, 1,048,576 bytes; `bee06776d07751d10e40bca0139e8d2a9aa108b8b1448f41109460b40c21bb2a` | `centroids.f32`: exact BF16-to-little-endian-F32 expansion, 2,097,152 bytes; `d293fc2fc2b68dea9716cc6cad81c4847084640d393962c637ef593415aa68c7` |
| `masked_embedding.token_ordering` | I64, shape 262144, 2,097,152 bytes; `21c3ab902cd1f15e6adf5389d196861a80831474e4963ad5f6284bc32b941f04` | `ordering.u32`: lossless conversion to little-endian U32 canonical token IDs, 1,048,576 bytes; `2d4a619b6fdf687972daaf298bf5ce341f4e2f1d05c20de10c77684e20481b07` |

The original retrieval selected this exact revision to match the assistant GGUF. Its base-model link names `google/gemma-4-E4B-it-assistant`; that link does not replace the QAT source revision above. The runtime authenticates both embedded assets before owner-local upload. Bundling them does not admit a quality or performance profile.
