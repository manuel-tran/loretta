# LoReTTa @ NeurIPS 2023
Official repository for LoReTTa ($`\textbf{L}`$inking m$`\textbf{O}`$dalities with a t$`\textbf{R}`$ansitive and commutativ$`\textbf{E}`$ pre-$`\textbf{T}`$raining s$`\textbf{T}`$r$`\textbf{A}`$tegy). [[arXiv](https://arxiv.org/abs/2305.14243)] [[website](https://nips.cc/virtual/2023/poster/70522)]


While we regret that we cannot release the full code due to company policy, we do our best to guide researchers in implementing our work. We provide the dataset, pseudocode, and point to implementations of related work. 

## Method

Imagine we have two datasets, one with paired image and text, and one with paired text and audio. How do we train a multimodal model that also works with paired image and audio? Here we introduce commutative and transitive pre-training (see also pseudocode.py): 

1. Train a model to generate image from text, text from image, text from audio, and audio from text.
2. Given a paired sample (image, text), we use the text to generate audio as a pseudo data point.
3. The generated audio, aligned with the text, is then used as conditioning to generate an image.
4. The generated image is compared to the original image in (image, text) to enforce consistency.
5. This is how we connect image and audio. This also works the other way around with (text, audio).

## Models

LoReTTa is a self-supervised learning framework that works with any modality-agnostic architecture. We choose the Transformer decoder for its simplicity and scalability. For the best performance, we recommend using its modern implementation based on [Llama](https://github.com/facebookresearch/llama/blob/main/llama/model.py) or [Mistral](https://github.com/mistralai/mistral-src/blob/main/mistral/model.py). We also enable [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) to speed up training and inference time. Alternative models that can handle sequences like [Hyena](https://github.com/HazyResearch/flash-fft-conv) or [Mamba](https://github.com/state-spaces/mamba) can also be used.

## Tokenization

The input to the Transformer is a sequence of tokens. So we need to tokenize our data. For images, we use image patches as tokens; for text, we use subwords as tokens; and so on. Since we are modeling the data in pixel space, we can either use the raw discretized values or pre-trained [VQ-VAEs](https://github.com/openai/DALL-E). It is also possible to model the data in [latent space](https://arxiv.org/abs/2309.17080) to avoid using VQ-VAEs. In svl_mnist.py, we show an example using the byte values as tokens.

## Causal modeling

The core of LoReTTa is next token prediction (also known as causal language modeling). It is currently one of the most powerful frameworks for generative pre-training due to its data efficiency, as training can be effectively parallelized using attention masks. During training the input and target are shifted by [one](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py#L165) and a [upper-triangular causal attention mask](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L63) is used so that only the previous tokens can be used to predict the next one.

## Multimodality

Since language modeling only models the next token given previous tokens, these tokens can theoretically come from any modality -- in any order. This idea is explored in [DALLE](https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/dalle_pytorch.py#L576) and [MMGPT](https://github.com/mugen-org/MUGEN_baseline/blob/eb0c35b82a1cc3058bbe364f59a423294fb59e20/lib/models/gpt/gpt.py#L109) to generate images from text and more. In a nutshell, these methods model the relation $`A \rightarrow B`$. We go one step further and model $`B \rightarrow A`$ as well. In fact, if the model has enough capacity, it can handle even more modality combinations, such as $`B \rightarrow C`$ and $`B \rightarrow A`$. To help the model better distinguish between different modalities, we prepend a class token (or modality token) to each modality.
