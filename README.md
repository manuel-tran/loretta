# LoReTTa @ NeurIPS 2023
Official repository for LoReTTa ($`\textbf{L}`$inking m$`\textbf{O}`$dalities with a t$`\textbf{R}`$ansitive and commutativ$`\textbf{E}`$ pre-$`\textbf{T}`$raining s$`\textbf{T}`$r$`\textbf{A}`$tegy). [[arXiv](https://arxiv.org/abs/2305.14243)] [[website](https://nips.cc/virtual/2023/poster/70522)]


While we regret that we cannot release the full code due to internal policy, we do our best to guide interested researchers in reproducing or implementing our work.

## Model architecture

Our multimodal model is based on the modern implementation of the Transformer decoder, most notably [LLama](https://github.com/facebookresearch/llama/blob/main/llama/model.py) and [Mistral](https://github.com/mistralai/mistral-src/blob/main/mistral/model.py). We recommend enabling [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) to speed up training and inference time. Alternative models that can process sequences such as [Hyena](https://github.com/HazyResearch/flash-fft-conv) or [Mamba](https://github.com/state-spaces/mamba) can also be used.

## Causal modeling

The core of LoReTTa is next token prediction (also known as causal language modeling). Currently, it is the most popular framework for generative pre-training. During training the input and target are shifted by [one](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/pretrain/tinyllama.py#L165) and a [upper-triangular causal attention mask](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L63) is used such that only the previous tokens can be used to generate the next.

## Multimodal modeling

We see causal modeling with Transformers as a modality agnostic framework. Instead of predicting the next token from the same modality, one can predict the next token from other modalities as well. Models that use this idea are [DALLE](https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/dalle_pytorch.py#L576) and [MMGPT](https://github.com/mugen-org/MUGEN_baseline/blob/eb0c35b82a1cc3058bbe364f59a423294fb59e20/lib/models/gpt/gpt.py#L109). They model $`A \rightarrow B`$ while we go one step further and model $`B \rightarrow A`$ as well. In fact, if the model has enough capacity, it can process even more modality combinations, for example, $`B \rightarrow C`$ and $`B \rightarrow A`$. In practise, we prepend a class token (or modality token) to every modality to help the model telling the modalities apart.
