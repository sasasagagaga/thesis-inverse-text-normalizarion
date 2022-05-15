
# Pytorch
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..metrics import compute_cls_metrics, join_results

# Models
from .transformer_core import PositionalEncoding, TokenEmbedding, create_mask
from .core import train_and_validate_epoch, validate_epoch


class TransformerEncoderClassifier(nn.Module):
    name_for_saving = "transformer_encoder_classifier"
    params_to_save = (
        "src_vocab_size",
        "tgt_vocab_size",
        "emb_size",
        "nhead",
        "num_encoder_layers",
        "dim_feedforward",
        "dropout",
    )

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            emb_size: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 512,
            dropout: float = 0.1
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor,
            src_padding_mask: torch.Tensor,
    ):
        memory = self.encode(src, src_mask, src_padding_mask)
        return self.generator(memory)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer_encoder(src_emb, src_mask, src_padding_mask)


def CLI_transformer_encoder_cls(src, tgt, model, tokenizer, device, loss_fn):
    src_mask, src_padding_mask = create_mask(tokenizer, src)
    logits = model(src, src_mask, src_padding_mask)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    return loss


def transformer_encoder_cls_translate(model, tokenizer, device, srcs):
    model = model.to(device)
    model.eval()

    if isinstance(srcs, torch.Tensor):
        tokens = srcs
    else:
        if not isinstance(srcs, list):
            srcs = [srcs]
        batch = [torch.tensor(tokenizer(src)["input_ids"]) for src in srcs]
        tokens = pad_sequence(batch, padding_value=tokenizer.pad_token_id)

    src = tokens.long()
    src_mask, src_padding_mask = create_mask(tokenizer, src)

    outs = model(src, src_mask, src_padding_mask)
    return outs


def TaVE_transformer_encoder_cls(
        model_version,
        epoch,
        model,
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer, loss_fn,
        writer, compute_metrics_every_iters=2000, print_every_iters=1000, write_every_iters=200, train_val_ratio=4
):
    def calc_loss_clb(src, tgt):
        return CLI_transformer_encoder_cls(src, tgt, model, tokenizer, device, loss_fn)

    train_and_validate_epoch(
        model_version,
        epoch,
        model, calc_loss_clb, compute_cls_metrics, transformer_encoder_cls_translate, join_results, "all",
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer,
        writer, compute_metrics_every_iters, print_every_iters, write_every_iters, train_val_ratio
    )


def VE_transformer_encoder_cls(epoch, model, tokenizer, device, dataloader_val, loss_fn, writer):
    def calc_loss_clb(src, tgt):
        return CLI_transformer_encoder_cls(src, tgt, model, tokenizer, device, loss_fn)

    validate_epoch(
        epoch,
        model, calc_loss_clb, compute_cls_metrics, transformer_encoder_cls_translate, join_results, "all",
        tokenizer, device,
        dataloader_val,
        writer
    )
