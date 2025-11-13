from multiprocessing import Value
from pathlib import Path
import json
from safetensors.numpy import load
from torch._functorch.config import max_dist_from_bw
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from safetensors.torch import load_file

from transformers.generation.utils import TopKLogitsWarper
from transformers.models.whisper.generation_whisper import LogitsProcessorList
from transformers.pytorch_utils import Conv1D

"""
{
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}
"""


@dataclass
class MyGPT2ModelConfig:
    N_HEAD: int
    N_LAYER: int
    VOCAB_SIZE: int
    N_EMBED: int
    ACTIVATION_FUNCTION: str
    ATTN_PDROP: float
    N_CONTEXT: int
    N_POSITIONS: int
    LAYER_NORM_EPSILON: float
    RESID_PDROP: float
    EMBD_PDROP: float

    # init function loads from a given config file
    def __init__(self, config_file: Path):
        super().__init__()
        with open(config_file, "r") as f:
            config = json.load(f)
        self.N_HEAD = config["n_head"]
        self.N_LAYER = config["n_layer"]
        self.VOCAB_SIZE = config["vocab_size"]
        self.N_EMBED = config["n_embd"]
        self.ACTIVATION_FUNCTION = config["activation_function"]
        self.N_CONTEXT = config["n_ctx"]
        self.N_POSITIONS = config["n_positions"]
        self.LAYER_NORM_EPSILON = config["layer_norm_epsilon"]
        self.ATTN_PDROP = config["attn_pdrop"]
        self.RESID_PDROP = config["resid_pdrop"]
        self.EMBD_PDROP = config["embd_pdrop"]


class MyGPT2ModelMultiHeadAttention(nn.Module):
    def __init__(self, config: MyGPT2ModelConfig):
        super(MyGPT2ModelMultiHeadAttention, self).__init__()
        self.input_dim_ =config.N_EMBED
        self.num_heads_ = config.N_HEAD
        self.head_dim_ = self.input_dim_ // self.num_heads_
        # Opposite of Linear Layer:
        # Conv1D(nf, nx) = Linear(nx, nf)
        # [input_dim] -> [Q | K | V] where each of Q,K,V have input_dim
        # This is just compact storage format
        # We will do the computation in a such a way that
        # Each head will have its own Q,K,V
        self.c_attn = Conv1D(self.input_dim_* 3,self.input_dim_)
        self.c_proj = Conv1D(self.input_dim_,self.input_dim_)

        # Dunno why we need it as of now
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.N_POSITIONS, config.N_POSITIONS), dtype=torch.bool)).view(
                1, 1, config.N_POSITIONS, config.N_POSITIONS
            ),
            persistent=True,
        )

    """
	[N, N_EMBED] -> [N, N_EMBED]
	"""

    def forward(self, x):
        # Sequence Length, Embedding Dimension
        S, D = x.shape
        # Getting packed Q, K, V
        # [S, D] -> [S, D *3]
        # [5, 768] -> [5, 2304]
        qkv = self.c_attn(x)

        # Now separate Q, K, V
        # [S, D * 3] -> [S, D], [S, D], [S, D]
        Q, K, V = qkv.split(D, dim=-1)

        def split_head(X: torch.Tensor):
            # [S, D]
            # [S, N_HEAD, HEAD_DIM]
            # [N_HEAD, S, HEAD_DIM]
            # return X.reshape(S, self.num_heads_, self.head_dim_).
            return X.view(S, self.num_heads_, self.head_dim_).transpose(-3, -2)

        # [N_HEAD, S, HEAD_DIM]
        Q = split_head(Q)
        K = split_head(K)
        V = split_head(V)

        # Q * K_T / sqrt(d): [N_HEAD, S, HEAD_DIM] * [N_HEAD, HEAD_DIM, S] -> [N_HEAD, S, S]
        raw_score = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim_)
        # Masking
        # [S, S]
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        # Reshape for broadcasting
        mask = mask.view(1, S, S)
        # [N_HEAD, S, S]
        # Places which are masked should be -inf
        masked_score = raw_score.masked_fill(mask == False, float("-inf"))
        # Attention: [N_HEAD, S, S]
        attention = F.softmax(masked_score, dim=-1)
        # Weighted sum of values
        # A * V
        # [N_HEAD, S, S] * [N_HEAD, S, HEAD_DIM] -> [N_HEAD, S, HEAD_DIM]
        weighted_sum = torch.matmul(attention, V)

        def merge_heads(X):
            # Combine weighted sum
            # [N_HEAD, S, HEAD_DIM]
            # -> [S, N_HEAD, HEAD_DIM]
            # -> [S, D]
            return X.transpose(-3, -2).reshape(S, D)

        combined_head_result = merge_heads(weighted_sum)

        # Final MLP layer which refines the attention output
        return self.c_proj(combined_head_result)


class MyGPT2ModelFeedForward(nn.Module):
    def __init__(self, input_dim: int):
        super(MyGPT2ModelFeedForward, self).__init__()
        self.input_dim_ = input_dim
        # IMPORTANT: We use 4 times the input dimension for the output dimension
        self.output_dim_ = input_dim * 4
        self.c_proj = Conv1D(self.input_dim_, self.output_dim_)
        self.c_fc = Conv1D(self.output_dim_, self.input_dim_)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.activation(self.c_fc(x)))


class MyGPT2ModelTranformerBlock(nn.Module):
    """
    A transformer block combines a few things:
            - Layer Norm
            - Masked Multi head attention
                    - Each head dimension is `N_EMBED // N_HEAD`
            - Residual connection
            - MLP
    """

    def __init__(self, config: MyGPT2ModelConfig):
        super(MyGPT2ModelTranformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.N_EMBED, eps=config.LAYER_NORM_EPSILON)
        self.attn = MyGPT2ModelMultiHeadAttention(config)
        self.dropout1 = nn.Dropout(config.ATTN_PDROP)
        self.ln_2 = nn.LayerNorm(config.N_EMBED, eps=config.LAYER_NORM_EPSILON)
        self.mlp = MyGPT2ModelFeedForward(config.N_EMBED)
        self.dropout2 = nn.Dropout(config.ATTN_PDROP)
        self.ln = [self.ln_1, self.ln_2]

    def forward(self, x):
        """
        [N, N_EMBED] -> [N, HEAD_DIM x N_HEAD] == [N, N_EMBED]
        where HEAD_DIM = N_EMBED // N_HEAD
        """
        y = self.ln_1(x)
        y = self.attn(y)
        y = self.dropout1(y)
        # residual
        y = x + y
        z = self.ln_2(y)
        z = self.mlp(z)
        z = self.dropout2(z)
        # residual
        return y + z


class MyGPT2ModelTransformer(nn.Module):
    def __init__(self, config: MyGPT2ModelConfig):
        super(MyGPT2ModelTransformer, self).__init__()
        self.config_ = config
        self.wte = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        self.wpe = nn.Embedding(config.N_POSITIONS, config.N_EMBED)
        self.drop = nn.Dropout(config.RESID_PDROP)
        self.h = nn.ModuleList(
            [MyGPT2ModelTranformerBlock(config) for _ in range(config.N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(config.N_EMBED, eps=config.LAYER_NORM_EPSILON)



    def forward(self, x):
        token_embed = self.wte(x)
        position_embed = self.wpe(torch.arange(x.size(0)))
        x = token_embed + position_embed

        # droput on input
        x = self.drop(x)
        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        return x


class MyGPT2Model(nn.Module):
    def __init__(self, config: MyGPT2ModelConfig):
        super(MyGPT2Model, self).__init__()
        self.transformer = MyGPT2ModelTransformer(config)
        self.lm_head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)

    def forward(self, x):
        x = self.transformer(x)
        x = self.lm_head(x)
        return x


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    # Set one only
    if top_k and top_p:
        raise ValueError("Please set only one of top_k or top_p")

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    if top_k:
        # return the indices of top k indices
        sample_probs, sample_indices = torch.topk(probs, k=top_k)
        # normalize the probabilities
        sample_probs = sample_probs / sample_probs.sum()
    elif top_p:
        # return the set whose total probability is greater than or equal to top_p
        # sort probability and indices together
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_indices = cumulative_probs < top_p
        sample_indices = sorted_indices[keep_indices]
        sample_probs = sorted_probs[keep_indices]
        # normalise the probabilities
        sample_probs = sample_probs / sample_probs.sum()
    else:
        raise ValueError("Please set one of top_k or top_p")

    # sample from sample_indices
    return sample_indices[torch.multinomial(sample_probs, num_samples=1)]


def generate(model, tokenizer, input_txt, max_length=10, top_k=None, top_p=None):
    tokens = tokenizer(input_txt, return_tensors="pt")["input_ids"][0]
    for _ in range(len(tokens), max_length):
        logits = model(tokens)
        last_logit = logits[-1]
        predicted_token_id = sample(last_logit, top_k=top_k, top_p=top_p)
        tokens = torch.cat([tokens, predicted_token_id], dim=-1)
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    # load from gpt2_model directory
    torch.set_grad_enabled(False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2_model", local_files_only=True)
    torch.set_printoptions(precision=2)
    gpt2_config = MyGPT2ModelConfig(Path("./gpt2_model/config.json"))
    print(gpt2_config)
    gpt2 = MyGPT2Model(config=gpt2_config)
    state_dict = load_file(Path("./gpt2_model/model.safetensors"))
    gpt2.transformer.load_state_dict(state_dict)
    # The lm head weight and token embedding weight are same
    gpt2.lm_head.weight = nn.Parameter(state_dict["wte.weight"])
    gpt2.eval()
    input_txt = "The capital of India is New"
    output_txt = generate(gpt2, tokenizer, input_txt, max_length=10, top_k=5)
    print("Input: ", input_txt)
    print("Output: ", output_txt)

    # Compare with original implementation

    # model = GPT2LMHeadModel.from_pretrained("gpt2_model", local_files_only=True)
    # tokens = tokenizer(input_txt, return_tensors="pt")
    # top_k_logits_wrapper = TopKLogitsWarper(top_k=5)
    # logits_processor = LogitsProcessorList([top_k_logits_wrapper])
    # result = model.generate(**tokens, max_length=10, logits_processor=logits_processor)
    # result_text = tokenizer.decode(result[0])
    # print(result_text)
    # print(f"Our Output: {output_txt}\nOriginal Model Output: {result_text}")
