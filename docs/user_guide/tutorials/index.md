# Tutorials

Step-by-step tutorials to guide you through complete workflows, from data preparation to serving trained models in production.

## [Train a Speculator](train.md)

The main end-to-end walkthrough: prepare data, generate hidden states, train, and serve. Covers Eagle-3, P-EAGLE, DFlash, DSpark, and MTP, in online, offline, or hybrid mode -- pick your algorithm and mode at the top of the page.

## [Train Eagle-3 for Gemma-4-31B-it (Online, Disaggregated)](train_eagle3_online_gemma4_31b.md)

End-to-end Gemma-4-31B-it recipe, including running the CUDA-13 vLLM stack on an older (CUDA 12.7) GPU node via forward compatibility, single-GPU fallback, and the dependency/config fixes needed.

**Time required:** ~1 hour (single A100 80 GB, 20k samples)

## [Response Regeneration](response_regeneration.md)

Regenerate dataset responses using your target model for improved drafter alignment. Recommended before training.

## [Evaluating Model Performance](evaluating_performance.md)

Benchmark and evaluate your trained speculator models.

## [GLM-5.2 MTP Evaluation Results](glm52_mtp_results.md)

Published GLM-5.2 native MTP evaluation results.

## [Serve in vLLM](serve_vllm.md)

Deploy your trained speculator models in vLLM for production inference.
