# LoReTTa
Training Transitive and Commutative Multimodal Transformers with LoReTTa (NeurIPS 2023)

## Overview

This is the official repository for the multimodal learning paradigm LoReTTa ($`\textbf{L}`$inking m$`\textbf{O}`$dalities with a t$`\textbf{R}`$ansitive and commutativ$`\textbf{E}`$ pre-$`\textbf{T}`$raining s$`\textbf{T}`$r$`\textbf{A}`$tegy). While we regret that we cannot release the full code due to internal policy, we do our best to guide interested researchers in reproducing or implementing our work.

## Model architecture

Our multimodal model is based on the modern implementation of the Transformer decoder, most notably [LLama](https://github.com/facebookresearch/llama/blob/main/llama/model.py) and [Mistral](https://github.com/mistralai/mistral-src/blob/main/mistral/model.py). Alternative models that can process sequences such as [Hyena](https://github.com/HazyResearch/flash-fft-conv) or [Mamba](https://github.com/state-spaces/mamba) can also be used.

