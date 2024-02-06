# LoReTTa
Training Transitive and Commutative Multimodal Transformers with LoReTTa (NeurIPS 2023)

## Method overview

This is the official repository for the multimodal learning paradigm LoReTTa ($`\textbf{L}`$inking m$`\textbf{O}`$dalities with a t$`\textbf{R}`$ansitive and commutativ$`\textbf{E}`$ pre-$`\textbf{T}`$raining s$`\textbf{T}`$r$`\textbf{A}`$tegy). While we regret that we cannot release the full code due to internal policy, we do our best to guide interested researchers in reproducing or implementing our work.

## Model architecture

Our multimodal model is based on the modern implementation of the Transformer decoder, most notably [LLama](https://github.com/facebookresearch/llama/blob/main/llama/model.py) and [Mistral](https://github.com/mistralai/mistral-src/blob/main/mistral/model.py). We recommend enabling [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) to speed up training and inference time. Alternative models that can process sequences such as [Hyena](https://github.com/HazyResearch/flash-fft-conv) or [Mamba](https://github.com/state-spaces/mamba) can also be used.

## Causal modeling

The core of LoReTTa is next token prediction (also known as causal language modeling). Currently, it is the most popular framework for generative pre-training. During training the input and target are shifted by one and a upper-triangular causal attention mask is used such that only the previous tokens can be used to generate the next [one](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py#L165).

## Multimodal modeling
