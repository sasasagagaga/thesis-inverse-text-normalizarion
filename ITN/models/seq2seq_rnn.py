
# Common libraries
import random

# Pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Local imports
from .core import train_and_validate_epoch, validate_epoch
from ..tokenizer import get_tokenizer
from ..metrics import compute_asr_metrics, join_results


MAX_LENGTH = 128
TEACHER_FORCING_RATIO = 0.5


class EncoderRNN(nn.Module):
    def __init__(self, src_vocab_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, tgt_vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(tgt_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, input.shape[0], -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, tgt_vocab_size, num_layers, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.tgt_vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.tgt_vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, input.shape[0], -1)
        embedded = self.dropout(embedded)  # 1 x batch_size x hidden_size

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[-1]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1)).transpose(0, 1)

        output = torch.cat((embedded, attn_applied), 2)  # 1 x batch_size x 2*hidden_size
        output = self.attn_combine(output)  # 1 x batch_size x hidden_size

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden


class Seq2SeqRNN(nn.Module):
    name_for_saving = "rnn"
    params_to_save = (
        "tokenizer_name",
        "hidden_size",
        "num_layers",
        "dropout_p",
        "use_attn",
    )

    def __init__(
            self,
            tokenizer_name=None,  # Note that it's important to store tokenizers immutable on hard drive
            hidden_size=512,
            num_layers=2,
            dropout_p=0.1,
            use_attn=True
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(self.tokenizer_name)
        self.src_vocab_size = len(self.tokenizer)
        self.tgt_vocab_size = len(self.tokenizer)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.use_attn = use_attn

        self.encoder = EncoderRNN(self.src_vocab_size, hidden_size, num_layers)
        if use_attn:
            self.decoder = AttnDecoderRNN(hidden_size, self.tgt_vocab_size, num_layers, dropout_p)
        else:
            self.decoder = DecoderRNN(hidden_size, self.tgt_vocab_size, num_layers)

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    def init_weights(self):
        pass


def CLI_seq2seq_rnn(
    input_tensor, target_tensor,
    model,
    tokenizer, device,
    criterion, max_length=MAX_LENGTH
):
    batch_size = input_tensor.shape[1]
    target_length = len(target_tensor)

    sum_loss, cnt_loss = 0, 0

    encoder_outputs_, encoder_hidden = model.encoder(input_tensor, None)
    encoder_outputs = torch.zeros(max_length, batch_size, model.encoder.hidden_size, device=device)
    encoder_outputs[:len(encoder_outputs_)] = encoder_outputs_

    decoder_input = torch.full((batch_size,), tokenizer.bos_token_id, device=device)
    decoder_hidden = encoder_hidden

    if random.random() < TEACHER_FORCING_RATIO:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            sum_loss += criterion(decoder_output, target_tensor[di])
            cnt_loss += 1
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        eos_check = torch.zeros(batch_size, dtype=torch.bool)
        for di in range(target_length):
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            next_word = decoder_output.argmax(dim=-1).squeeze().detach()  # detach from history as input
            decoder_input = next_word

            eos_check[next_word == tokenizer.eos_token_id] = 1

            sum_loss += criterion(decoder_output, target_tensor[di])
            cnt_loss += 1
            if torch.all(eos_check):
                break

    avg_loss = sum_loss / cnt_loss
    return avg_loss


@torch.no_grad()
def seq2seq_rnn_translate(model, tokenizer, device, srcs, tgts=None, max_length=MAX_LENGTH):
    model.eval()

    if isinstance(srcs, torch.Tensor):
        tokens = srcs
    else:
        if not isinstance(srcs, list):
            srcs = [srcs]

        batch = [
            torch.tensor(
                [tokenizer.bos_token_id] +
                tokenizer(src)["input_ids"] +
                [tokenizer.eos_token_id]
            )
            for src in srcs
        ]

        tokens = pad_sequence(batch, padding_value=tokenizer.pad_token_id)

    input_tensor = tokens.to(device)

    batch_size = input_tensor.shape[1]

    encoder_outputs_, encoder_hidden = model.encoder(input_tensor, None)
    encoder_outputs = torch.zeros(max_length, batch_size, model.encoder.hidden_size, device=device)
    encoder_outputs[:len(encoder_outputs_)] = encoder_outputs_

    decoder_input = torch.full((batch_size,), tokenizer.bos_token_id, device=device)
    decoder_hidden = encoder_hidden

    result = []
    max_len = max_length
    eos_check = torch.zeros(batch_size, dtype=torch.bool)
    for cur_len in range(1, max_length + 1):
        decoder_output, decoder_hidden = model.decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        next_word = decoder_output.argmax(dim=-1)
        next_word[eos_check | (cur_len >= max_len)] = tokenizer.pad_token_id

        eos_check[next_word == tokenizer.eos_token_id] = 1

        result.append(next_word)
        if torch.all(eos_check):
            break

        decoder_input = next_word.squeeze()

    result = torch.vstack(result).T
    result = [tokenizer.decode(res_tokens, skip_special_tokens=True) for res_tokens in result]
    if tgts is not None:
        result = result, [tokenizer.decode(x, skip_special_tokens=True) for x in tgts.T]
    return result


def TaVE_seq2seq_rnn(
        model_version,
        epoch,
        model,
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer, loss_fn,
        writer, compute_metrics_every_iters=2000, print_every_iters=1000, write_every_iters=200, train_val_ratio=4
):
    def calc_loss_clb(src, tgt):
        return CLI_seq2seq_rnn(src, tgt, model, tokenizer, device, loss_fn)

    train_and_validate_epoch(
        model_version,
        epoch,
        model, calc_loss_clb, compute_asr_metrics, seq2seq_rnn_translate, join_results, "all",
        language, tokenizer, device,
        dataloader_train, dataloader_val,
        optimizer,
        writer, compute_metrics_every_iters, print_every_iters, write_every_iters, train_val_ratio
    )


@torch.no_grad()
def VE_seq2seq_rnn(epoch, model, tokenizer, device, dataloader_val, loss_fn, writer):
    def calc_loss_clb(src, tgt):
        return CLI_seq2seq_rnn(src, tgt, model, tokenizer, device, loss_fn)

    validate_epoch(
        epoch,
        model, calc_loss_clb, compute_asr_metrics, seq2seq_rnn_translate, join_results, "all",
        tokenizer, device,
        dataloader_val,
        writer
    )
