# Model scope and provenance

## Preserved detector

The default `tidewall.prompt-injection.1` profile uses:

| Asset | Identity |
| --- | --- |
| Base model/tokenizer | `Qwen/Qwen3-0.6B` |
| Base/tokenizer revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| LoRA adapter | `abedegno/prompt-injection-classifier-qwen3-0p6b` |
| Adapter revision | `5500c3bef53729ed4661b5cfae1b01adca96ca4c` |
| Score | `two_class_softmax_uncalibrated` |
| Serving contract | Schema version 1, one complete user prompt |

The model is not a sequence-classification-head model. It scores the next-token
logits for the pinned `unsafe` and `safe` class tokens using the preserved
Qwen chat-template recipe. The recipe's unsafe definition includes prompt
injection, jailbreaks and other unsafe requests; a binary result does not
identify a particular threat category. The score is not a calibrated probability,
severity, policy decision or evidence of a successful block.

`tidewall_lora/model-manifest.json` pins artifact sizes/hashes and the prompt,
tokenizer and class-token identities. `recipe-provenance.json` pins original
source commit `7665e9f91e059893a09649d95596d9da7a3f2aa1`, its SHA-256 and the
preserved function hashes. The original source is
[prompt-injection-lora](https://github.com/abedegno/prompt-injection-lora).

## Standalone extraction

The initial standalone source was extracted from the inference directory of
Tidewall's maintained integration candidate at commit
`0cf9093114738ace4089f20309e231f2c24cc92a`.
That candidate's preparation/runtime implementation originated at
`ba1051fc48fd89ba27de9770d50446e090d94831`.
Runtime Python files and model/provenance JSON were copied without modification.
Repository documentation and test paths were adapted for independent use.
Neither OpenAIDR Git history nor its server, sensor, dashboard or proprietary
hosted intelligence SDK is part of this standalone source.

Research helper modules remain in the package to preserve the existing library;
their presence does not certify a public benchmark, training-data separation or
calibration. Integration-level evaluation drivers and datasets are excluded.

## Optional ensemble

The optional Q8 ensemble profile preserves an implementation adapted from
[prompt-injection-ensemble](https://github.com/abedegno/prompt-injection-ensemble),
commit `b9bc67ae6109a3f7bd1c81f73e218a69acf60f36`. It averages two specialist
scores. This is experimental, remains uncalibrated and is not selected by
standalone state preparation. Its weights are not distributed by this package.

## Evaluation and distribution boundaries

Current evidence does not certify a training-disjoint test set or enterprise
false-positive rate. Representative human-labelled normal agent workflows
remain absent. Synthetic startup examples are functional checks, not quality
measurements. CPU/GPU performance and production GPU reliability are not
qualified by this preview. No default model, threshold or weights were changed.

The package is Apache-2.0; see LICENSE and NOTICE. Base/adapter notices are
retained by the pinned artifact staging process. Distribution of model weights
is separate from distribution of this source; this repository ships no weights.
