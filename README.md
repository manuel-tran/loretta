# LoReTTa @ NeurIPS 2023
Official repository for LoReTTa ($`\textbf{L}`$inking m$`\textbf{O}`$dalities with a t$`\textbf{R}`$ansitive and commutativ$`\textbf{E}`$ pre-$`\textbf{T}`$raining s$`\textbf{T}`$r$`\textbf{A}`$tegy). [[arXiv](https://arxiv.org/abs/2305.14243)] [[website](https://nips.cc/virtual/2023/poster/70522)]


While we regret that we cannot release the full code due to company policy, we do our best to guide researchers in implementing our work. We provide the dataset, pseudocode, and point to implementations of related work. 

## Method

Imagine we have two datasets, one with paired image and text, and one with paired text and audio. How do we train a multimodal model that also works with paired image and audio? Here we introduce commutative and transitive pre-training: 

1. Train a model to generate image from text, text from image, text from audio, and audio from text.
2. Given a paired sample (image, text), we use the text to generate audio as a pseudo data point.
3. The generated audio, aligned with the text, is then used as conditioning to generate an image.
4. The generated image is compared to the original image in (image, text) to enforce consistency.
5. This is how we connect image and audio. This also works the other way around with (text, audio).

## Models

LoReTTa is a self-supervised framework that can work with any modality-agnostic architecture. We choose the transformer decoder for its simplicity and scalability. For the best performance, we recommend using its modern implementation based on [Llama](https://github.com/facebookresearch/llama/blob/main/llama/model.py) and [Mistral](https://github.com/mistralai/mistral-src/blob/main/mistral/model.py). We also enable [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) to speed up training and inference time. Alternative models that can handle sequences like [Hyena](https://github.com/HazyResearch/flash-fft-conv) or [Mamba](https://github.com/state-spaces/mamba) can also be used.

## Causal modeling

The core of LoReTTa is next token prediction (also known as causal language modeling). Currently, it is the most popular framework for generative pre-training. During training the input and target are shifted by [one](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py#L165) and a [upper-triangular causal attention mask](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L63) is used such that only the previous tokens can be used to generate the next.

## Multimodal modeling

We see causal modeling with Transformers as a modality agnostic framework. Instead of predicting the next token from the same modality, one can predict the next token from other modalities as well. Models that use this idea are [DALLE](https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/dalle_pytorch.py#L576) and [MMGPT](https://github.com/mugen-org/MUGEN_baseline/blob/eb0c35b82a1cc3058bbe364f59a423294fb59e20/lib/models/gpt/gpt.py#L109). They model $`A \rightarrow B`$ while we go one step further and model $`B \rightarrow A`$ as well. In fact, if the model has enough capacity, it can process even more modality combinations, for example, $`B \rightarrow C`$ and $`B \rightarrow A`$. In practise, we prepend a class token (or modality token) to every modality to help the model telling the modalities apart.
