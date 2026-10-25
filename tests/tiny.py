"""Build tiny stand-ins for the real models so every script can be tested on a CPU in seconds.

The tiny Qwen3 has the real architecture (28 decoder layers, tied embeddings, q/k/v/o and
gate/up/down projections) but tiny widths and random weights. Its tokenizer is a small byte-level
BPE with Qwen's special tokens and a copy of Qwen3's chat template (including the empty
<think></think> block that enable_thinking=False inserts). Nothing here is used on Kaggle.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

QWEN3_CHAT_TEMPLATE = r"""{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- set ns = namespace(last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if message.role == "user" and ns.last_query_index == messages|length - 1 and index != messages|length - 1 or (message.role == "user" and loop.first) %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' + message.content }}
        {%- else %}
            {{- '<|im_start|>assistant\n' + message.content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]

CORPUS = [
    "Janet has 3 apples and buys 5 more. How many apples does she have? 3 + 5 = 8. The answer is 8.",
    "Q: A train travels 60 miles in 2 hours. What is its speed?\nA: 60 / 2 = 30 miles per hour.",
    "Hello! I can fulfill your one wish. What is the one thing you want? hey I feel bored",
    "A man is sitting on a roof. He starts pulling up roofing on the roof. Then he",
    "Does this image contain a dog? yes no How many cats can you see? two three four",
    "Write a short caption for this image. A group of people standing next to a red bus.",
    "Answer the question using a single word or phrase. What color is the shirt? blue",
    "Think step by step inside tags, then give your final answer inside \\boxed{42}.",
    "{\"answer\": \"I would like to travel\"} idea_1 = # Your first idea",
] * 20


def build_tokenizer(out: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>",
                                   pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIALS[1:])
    fast.chat_template = QWEN3_CHAT_TEMPLATE
    fast.save_pretrained(out)
    return fast


def build_qwen(out: Path, layers: int = 28):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = build_tokenizer(out)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True,
                      bos_token_id=None, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    # Make the weights less trivially random so SVDs and norms are well behaved.
    model.to(torch.bfloat16)  # the real checkpoint ships in BF16
    model.save_pretrained(out)
    return out


def build_siglip(out: Path):
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor

    cfg = SiglipVisionConfig(hidden_size=768, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, image_size=224, patch_size=16)
    torch.manual_seed(0)
    SiglipVisionModel(cfg).save_pretrained(out)
    SiglipImageProcessor(size={"height": 224, "width": 224}).save_pretrained(out)
    return out


if __name__ == "__main__":
    root = Path(os.environ.get("TINY_DIR", "/tmp/tiny"))
    build_qwen(root / "qwen")
    build_siglip(root / "siglip")
    print(json.dumps({"qwen": str(root / "qwen"), "siglip": str(root / "siglip")}))
