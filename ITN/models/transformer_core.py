
# Common libraries
import math
from functools import lru_cache

# Pytorch
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[0], :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(size, device):
    mask = ~torch.ones((size, size), dtype=torch.bool, device=device).tril()
    return mask


def generate_padding_mask(tokenizer, src: torch.Tensor):
    return (src == tokenizer.pad_token_id).T


def create_mask(tokenizer, src: torch.Tensor = None, tgt: torch.Tensor = None):
    assert not (src is None and tgt is None), "src or tgt should be specified"

    to_return = tuple()
    if src is not None:
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool, device=src.device)
        src_padding_mask = generate_padding_mask(tokenizer, src)
        to_return = to_return + (src_mask, src_padding_mask)

    if tgt is not None:
        tgt_seq_len = tgt.shape[0]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_padding_mask = generate_padding_mask(tokenizer, tgt)
        to_return = to_return + (tgt_mask, tgt_padding_mask)

    return to_return


def model_encode(model, src, tokenizer, device, return_src_lengths=False):
    model = model.to(device)
    src: torch.Tensor = src.to(device)
    src_mask, src_padding_mask = create_mask(tokenizer, src)

    encode_ret = model.encode(src, src_mask, src_padding_mask)
    if not return_src_lengths:
        return encode_ret

    src_lengths = (src != tokenizer.pad_token_id).sum(dim=0).cpu()
    return encode_ret, src_lengths


@torch.inference_mode()
def memory_greedy_decode(
        model, memory, memory_key_padding_mask,
        max_lengths, start_symbol, tokenizer, device
):
    model = model.to(device)

    batch_size = memory.shape[1]

    eos_check = torch.zeros(batch_size, dtype=torch.bool)
    result = torch.full((1, batch_size), start_symbol, dtype=torch.long)  # always on CPU, move to GPU when needed
    for cur_len in range(1, max(max_lengths)):
        cuda_result = result.to(device)
        tgt_mask, tgt_padding_mask = create_mask(tokenizer, tgt=cuda_result)

        out = model.decode(cuda_result, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        prob = model.generator(out[-1])
        next_word = torch.argmax(prob, dim=-1).cpu()  # .item()
        next_word[eos_check | (cur_len >= max_lengths)] = tokenizer.pad_token_id

        eos_check[next_word == tokenizer.eos_token_id] = 1

        result = torch.vstack([result, next_word])
        if torch.all(eos_check):
            break

    return result


@torch.inference_mode()
def memory_beam_search(
        model, memory, memory_key_padding_mask, max_lengths,
        start_symbol, tokenizer, device, beam_width
):
    model = model.to(device)

    # Меняем размеры memory и memory_key_padding_mask, чтобы оно совпадало с размером входа в декодер трансформера
    mem_seq_len, batch_size, mem_emb_size = memory.shape

    @lru_cache  # For reuse of computations
    def reshape_memory_tensors_for_beam_width(eff_beam_width):
        """It's expected for eff_beam_width to be 1 or `beam_width`. But it can be arbitrary..."""
        if eff_beam_width == 1:
            return memory, memory_key_padding_mask

        return (
            memory
                .view(mem_seq_len, batch_size, 1, mem_emb_size)
                .expand(mem_seq_len, batch_size, eff_beam_width, mem_emb_size)
                .reshape(mem_seq_len, -1, mem_emb_size),
            memory_key_padding_mask
                .reshape(batch_size, 1, mem_seq_len)
                .expand(batch_size, eff_beam_width, mem_seq_len)
                .reshape(-1, mem_seq_len)
        )

    eos_check = torch.zeros((batch_size, beam_width), dtype=torch.bool)

    # always on CPU, move to GPU when needed
    result = torch.full((1, batch_size, 1), start_symbol, dtype=torch.long)

    log_probs = torch.full((batch_size, 1), 0, dtype=torch.float64, device=device)

    for cur_len in range(1, max(max_lengths)):
        effective_beam_width = result.shape[2]
        cur_memory, cur_memory_key_padding_mask = reshape_memory_tensors_for_beam_width(effective_beam_width)
        cuda_result = result.view(cur_len, batch_size * effective_beam_width).to(device)  # -1, batch_size * beam_width
        tgt_mask, tgt_padding_mask = create_mask(tokenizer, tgt=cuda_result)

        out = model.decode(cuda_result, cur_memory, tgt_mask, tgt_padding_mask, cur_memory_key_padding_mask)
        logits = model.generator(out[-1]).view(batch_size, effective_beam_width, -1)

        next_log_probs = torch.log_softmax(logits, dim=-1)

        cur_log_probs = log_probs.unsqueeze(-1) + next_log_probs

        topk = torch.topk(cur_log_probs.view(batch_size, -1), beam_width, dim=-1)

        batch_ids = torch.arange(batch_size).view(-1, 1).expand(-1, beam_width)
        beam_ids = (torch.div(topk.indices, logits.shape[-1], rounding_mode='trunc')).cpu()
        next_words = (topk.indices % logits.shape[-1]).cpu()

        # Update EOS check with sequence lenghts limits
        eos_check |= (cur_len >= max_lengths).unsqueeze(1)

        # Restore beam_ids for finished sequences
        beam_ids[eos_check] = eos_check.argwhere().T[1]

        # Fill finished sequences with pad_token
        next_words[eos_check] = tokenizer.pad_token_id

        # Update EOS check with finished sequences
        eos_check[next_words == tokenizer.eos_token_id] = True

        result = torch.vstack([
            result[:, batch_ids, beam_ids],
            next_words.unsqueeze(0)
        ])

        log_probs = topk.values

        if torch.all(eos_check):
            break

    result = torch.select(result, dim=2, index=0)

    return result


def prepare_srcs_for_translate(srcs, tokenizer):
    if isinstance(srcs, torch.Tensor):
        tokens = srcs
    else:
        if not isinstance(srcs, list):
            srcs = [srcs]

        batch = [torch.tensor(
            [tokenizer.bos_token_id] + tokenizer(src)["input_ids"] + [tokenizer.eos_token_id]
        ) for src in srcs]

        tokens = pad_sequence(batch, padding_value=tokenizer.pad_token_id)

    src = tokens.long()

    return src
