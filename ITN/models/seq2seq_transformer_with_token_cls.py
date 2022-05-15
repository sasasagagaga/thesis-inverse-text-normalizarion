
# Numpy
import numpy as np

# Pytorch
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# Transformers
from transformers import get_linear_schedule_with_warmup

# Local imports
from ..tokenizer import get_tokenizer, add_class_tokens, add_range_class_tokens, decode
from ..data.core import class2ind, ind2class
from ..utils import make_fp_function

# Models
from .core import load_model
from .transformer_core import (
    PositionalEncoding, TokenEmbedding, create_mask,
    model_encode, memory_beam_search, memory_greedy_decode,
    prepare_srcs_for_translate
)
from .transformer_encoder_classifier import TransformerEncoderClassifier
from .seq2seq_transformer import Seq2SeqTransformer


class Seq2SeqTransformerWithEncoderForClasses(nn.Module):
    """
    Модель №1.
    For train use "TaVE_seq2seq_transformer" from "seq2seq_transformer" module.
    For inference use "VE_seq2seq_transformer", "seq2seq_transformer_greedy_decode"
    and "seq2seq_transformer_translate" from "seq2seq_transformer" module.
    """
    name_for_saving = "transformer_with_encoder_for_classes"
    params_to_save = (
        "transformer_encoder_params_for_load",
        "tokenizer_name",
        "emb_size",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
    )

    def __init__(
            self,
            transformer_encoder_params_for_load,
            tokenizer_name=None,  # Note that it's important to store tokenizers immutable on hard drive
            emb_size: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 512,
            dropout: float = 0.1
    ):
        super().__init__()

        self.transformer_encoder_params_for_load = transformer_encoder_params_for_load
        self.transformer_encoder_for_classes = load_model(
            TransformerEncoderClassifier,
            device="cpu",  # To load it on CPU-only machines
            **transformer_encoder_params_for_load
        )[0]
        for p in self.transformer_encoder_for_classes.parameters():
            p.requires_grad_(False)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(self.tokenizer_name)
        self.src_vocab_size = len(self.tokenizer)
        self.tgt_vocab_size = len(self.tokenizer)

        self.emb_size = emb_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_emb_size = emb_size + self.transformer_encoder_for_classes.emb_size
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(decoder_emb_size, self.tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(self.src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(self.tgt_vocab_size, decoder_emb_size)
        self.src_positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.tgt_positional_encoding = PositionalEncoding(decoder_emb_size, dropout=dropout)

    def forward(
            self,
            src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
    ):
        memory, memory_key_padding_mask = self.encode(src, src_mask, src_padding_mask)
        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        src_emb = self.src_positional_encoding(self.src_tok_emb(src))

        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        class_memory = self.transformer_encoder_for_classes.encode(src, src_mask, src_padding_mask)
        memory = torch.cat([memory, class_memory], dim=-1)

        return memory, src_padding_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
               tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.tgt_positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)

    def configure_optimizer(self, lr=1e-4):
        print(lr)
        return torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    @staticmethod
    def configure_lr_scheduler(optimizer, **params):
        return get_linear_schedule_with_warmup(optimizer, **params)

    def init_weights(self):
        for name, p in self.named_parameters():
            if "transformer_encoder_for_classes" not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Seq2SeqTransformerWithTokensClassifierBase(Seq2SeqTransformer):
    name_for_saving = "transformer_with_tokens_classifier_base"
    params_to_save = (
        "transformer_encoder_params_for_load",
    ) + Seq2SeqTransformer.params_to_save

    def __init__(self, transformer_encoder_params_for_load, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transformer_encoder_params_for_load = transformer_encoder_params_for_load
        self.transformer_encoder_for_classes = load_model(
            TransformerEncoderClassifier,
            device="cpu",  # To load it on CPU-only machines
            **transformer_encoder_params_for_load
        )[0]
        for p in self.transformer_encoder_for_classes.parameters():
            p.requires_grad_(False)

        self._add_necessary_tokens()
        self._resize_src_vocab_size_to_tokenizer_size_unsafe()  # Important fix!

    def _add_necessary_tokens(self):
        raise NotImplementedError

    def init_weights(self):
        for name, p in self.named_parameters():
            if "transformer_encoder_for_classes" not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Seq2SeqTransformerWithTokensClassLabelingBase(Seq2SeqTransformerWithTokensClassifierBase):
    name_for_saving = "transformer_with_tokens_class_labeling_base"

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса для токенов, для которых предсказанный класс не равен PLAIN"""
        raise NotImplementedError

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        src, src_mask, src_padding_mask = self._prepare_batch(src, src_mask, src_padding_mask)
        return super().encode(src, src_mask, src_padding_mask)[0], src_padding_mask


class Seq2SeqTransformerWithTokensClassLabelingTrain(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №2. Часть для обучения"""
    name_for_saving = "transformer_with_tokens_class_labeling"

    def _add_necessary_tokens(self):
        add_class_tokens(self.tokenizer)


class Seq2SeqTransformerWithTokensClassLabelingEval(Seq2SeqTransformerWithTokensClassLabelingBase):
    """Модель №2. Часть для инференса"""
    name_for_saving = "transformer_with_tokens_class_labeling"

    def _add_necessary_tokens(self):
        add_class_tokens(self.tokenizer)

    def _class2token(self, cur_class):
        token_for_group = ind2class[cur_class]
        token_for_group = f"[{token_for_group}]"
        token_for_group = self.tokenizer(token_for_group)["input_ids"]
        if len(token_for_group) > 1:
            raise ValueError("Strange token for class (after tokenization has more then 1 token)")
        return token_for_group[0]

    def _check_token_is_plain(self, token_class):
        return token_class in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса перед теми токенами, для которых предсказанный класс не равен PLAIN"""
        logits = self.transformer_encoder_for_classes(src, src_mask, src_padding_mask)
        batch_classes = logits.argmax(dim=-1).cpu()

        special_tokens_mask = (
                (src == self.tokenizer.bos_token_id) |
                (src == self.tokenizer.eos_token_id) |
                src_padding_mask.T
        )

        batch_classes[special_tokens_mask] = class2ind["PLAIN"]
        batch_classes = batch_classes.T.numpy()

        batch_tokens = src.T.cpu().numpy()
        batch_pad_mask = src_padding_mask.cpu().numpy()

        tokens_with_classes = []
        for classes, tokens, pad_mask in zip(batch_classes, batch_tokens, batch_pad_mask):
            new_tokens = []
            for cl, token, is_padding in zip(classes, tokens, pad_mask):
                if is_padding:
                    continue
                if not self._check_token_is_plain(cl):
                    new_tokens.append(self._class2token(cl))
                new_tokens.append(token)
            new_tokens = torch.tensor(new_tokens, dtype=torch.long)
            tokens_with_classes.append(new_tokens)

        tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=self.tokenizer.pad_token_id)
        new_src = tokens_with_classes.to(src.device)

        new_src_mask, new_src_padding_mask = create_mask(self.tokenizer, new_src)

        return new_src, new_src_mask, new_src_padding_mask


class Seq2SeqTransformerWithTokensClassRangeLabelingTrain(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №3. Часть для обучения"""
    name_for_saving = "transformer_with_tokens_class_range_labeling"

    def _add_necessary_tokens(self):
        add_range_class_tokens(self.tokenizer)


class Seq2SeqTransformerWithTokensClassRangeLabelingEval(Seq2SeqTransformerWithTokensClassLabelingBase):
    """Модель №3. Часть для инференса"""
    name_for_saving = "transformer_with_tokens_class_range_labeling"

    def _add_necessary_tokens(self):
        add_range_class_tokens(self.tokenizer)

    def _class2range_tokens(self, cur_class):
        cur_class = ind2class[cur_class]
        tokens_for_group = [f"[{cur_class}]", f"[/{cur_class}]"]
        begin_group_token, end_group_token = self.tokenizer(tokens_for_group)["input_ids"]
        if len(begin_group_token) != 1:
            raise ValueError("Strange begin group token for class (after tokenization has not 1 token)")
        if len(end_group_token) != 1:
            raise ValueError("Strange end group token for class (after tokenization has not 1 token)")
        return begin_group_token[0], end_group_token[0]

    def _check_token_is_plain(self, token_class):
        return token_class in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса вокруг тех групп токенов, для которых предсказанный класс не равен PLAIN"""
        logits = self.transformer_encoder_for_classes(src, src_mask, src_padding_mask)
        batch_classes = logits.argmax(dim=-1).cpu()

        special_tokens_mask = (
                (src == self.tokenizer.bos_token_id) |
                (src == self.tokenizer.eos_token_id) |
                src_padding_mask.T
        )

        batch_classes[special_tokens_mask] = class2ind["PLAIN"]
        batch_classes = batch_classes.T.numpy()

        batch_tokens = src.T.cpu().numpy()
        batch_pad_mask = src_padding_mask.cpu().numpy()

        tokens_with_classes = []
        for classes, tokens, pad_mask in zip(batch_classes, batch_tokens, batch_pad_mask):
            sequential_mask = classes[1:] != classes[:-1]
            sequential_mask = np.r_[True, sequential_mask, True]
            idx = np.where(sequential_mask)[0]

            new_tokens = []
            for begin, end in zip(idx, idx[1:]):
                if pad_mask[begin]:
                    continue
                if self._check_token_is_plain(classes[begin]):
                    new_tokens += tokens[begin:end].tolist()
                    continue

                begin_group_token, end_group_token = self._class2range_tokens(classes[begin])
                new_tokens += [begin_group_token] + tokens[begin:end].tolist() + [end_group_token]

            new_tokens = torch.tensor(new_tokens, dtype=torch.long)
            tokens_with_classes.append(new_tokens)

        tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=self.tokenizer.pad_token_id)
        new_src = tokens_with_classes.to(src.device)

        new_src_mask, new_src_padding_mask = create_mask(self.tokenizer, new_src)

        return new_src, new_src_mask, new_src_padding_mask


class Seq2SeqTransformerWithClassLabelingGroupDecoderTrain(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №4 — часть для обучения (на UTC — Using True Classes)"""
    name_for_saving = "transformer_with_class_labeling_group_decoder"

    def _add_necessary_tokens(self):
        add_class_tokens(self.tokenizer)


class Seq2SeqTransformerWithClassLabelingGroupDecoderEval(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №4 — часть для инференса (по сути UPC — Using Predicted Classes)"""
    name_for_saving = "transformer_with_class_labeling_group_decoder"

    def _add_necessary_tokens(self):
        add_class_tokens(self.tokenizer)

    def _class2token(self, cur_class):
        token_for_group = ind2class[cur_class]
        token_for_group = f"[{token_for_group}]"
        token_for_group = self.tokenizer(token_for_group)["input_ids"]
        if len(token_for_group) > 1:
            raise ValueError("Strange token for class (after tokenization has more then 1 token)")
        return token_for_group[0]

    def _check_token_is_plain(self, token):
        return token in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса для токенов, для которых предсказанный класс не равен PLAIN"""
        logits = self.transformer_encoder_for_classes(src, src_mask, src_padding_mask)
        batch_classes = logits.argmax(dim=-1).cpu()

        special_tokens_mask = (
                (src == self.tokenizer.bos_token_id) |
                (src == self.tokenizer.eos_token_id) |
                src_padding_mask.T
        )

        batch_classes[special_tokens_mask] = class2ind["PLAIN"]
        batch_classes = batch_classes.T.numpy()

        batch_tokens = src.T.cpu().numpy()
        batch_pad_mask = src_padding_mask.cpu().numpy()

        tokens_with_classes = []
        plain_groups = []
        for classes, tokens, pad_mask in zip(batch_classes, batch_tokens, batch_pad_mask):
            classes = classes[~pad_mask]
            tokens = tokens[~pad_mask]

            sequential_mask = classes[1:] != classes[:-1]
            sequential_mask = np.r_[True, sequential_mask, True]
            idx = np.where(sequential_mask)[0]

            cur_plain_groups = [[]]

            for begin, end in zip(idx, idx[1:]):
                if self._check_token_is_plain(classes[begin]):
                    if begin == 0 or self._check_token_is_plain(classes[begin - 1]):
                        cur_plain_groups[-1] += tokens[begin:end].tolist()
                    else:
                        cur_plain_groups.append(tokens[begin:end].tolist())
                    continue
                if end >= len(classes) or not self._check_token_is_plain(classes[end]):
                    cur_plain_groups.append([])

                class_token = self._class2token(classes[begin])
                mid_tokens = np.empty(2 * (end - begin), dtype=tokens.dtype)
                mid_tokens[0::2] = class_token
                mid_tokens[1::2] = tokens[begin:end]

                cur_tokens = np.r_[tokens[:begin], mid_tokens, tokens[end:]]

                tokens_with_classes.append(torch.tensor(cur_tokens, dtype=torch.long))

            plain_groups.append(cur_plain_groups)

        if len(tokens_with_classes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), plain_groups

        tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=self.tokenizer.pad_token_id)
        new_src = tokens_with_classes.to(src.device)

        new_src_mask, new_src_padding_mask = create_mask(self.tokenizer, new_src)

        return new_src, new_src_mask, new_src_padding_mask, plain_groups

    def forward(
            self,
            src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
    ):
        memory, memory_key_padding_mask, plain_groups = self.encode(src, src_mask, src_padding_mask)
        if len(memory) == 0:
            return memory, plain_groups

        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs), plain_groups

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        src, src_mask, src_padding_mask, plain_groups = self._prepare_batch(src, src_mask, src_padding_mask)
        if len(src) == 0:
            return src, src_padding_mask, plain_groups

        new_src = super().encode(src, src_mask, src_padding_mask)[0]
        return new_src, src_padding_mask, plain_groups


class Seq2SeqTransformerWithClassRangeLabelingGroupDecoderTrain(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №5 — часть для обучения"""
    name_for_saving = "transformer_with_class_range_labeling_group_decoder"

    def _add_necessary_tokens(self):
        add_range_class_tokens(self.tokenizer)


class Seq2SeqTransformerWithClassRangeLabelingGroupDecoderEval(Seq2SeqTransformerWithTokensClassifierBase):
    """Модель №5 — часть для инференса"""
    name_for_saving = "transformer_with_class_range_labeling_group_decoder"

    def _add_necessary_tokens(self):
        add_range_class_tokens(self.tokenizer)

    def _class2range_tokens(self, cur_class):
        cur_class = ind2class[cur_class]
        tokens_for_group = [f"[{cur_class}]", f"[/{cur_class}]"]
        begin_group_token, end_group_token = self.tokenizer(tokens_for_group)["input_ids"]
        if len(begin_group_token) != 1:
            raise ValueError("Strange begin group token for class (after tokenization has not 1 token)")
        if len(end_group_token) != 1:
            raise ValueError("Strange end group token for class (after tokenization has not 1 token)")
        return begin_group_token[0], end_group_token[0]

    def _check_token_is_plain(self, token):
        return token in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса для токенов, для которых предсказанный класс не равен PLAIN"""
        logits = self.transformer_encoder_for_classes(src, src_mask, src_padding_mask)
        batch_classes = logits.argmax(dim=-1).cpu()

        special_tokens_mask = (
                (src == self.tokenizer.bos_token_id) |
                (src == self.tokenizer.eos_token_id) |
                src_padding_mask.T
        )

        batch_classes[special_tokens_mask] = class2ind["PLAIN"]
        batch_classes = batch_classes.T.numpy()

        batch_tokens = src.T.cpu().numpy()
        batch_pad_mask = src_padding_mask.cpu().numpy()

        tokens_with_classes = []
        plain_groups = []
        for classes, tokens, pad_mask in zip(batch_classes, batch_tokens, batch_pad_mask):
            classes = classes[~pad_mask]
            tokens = tokens[~pad_mask]

            sequential_mask = classes[1:] != classes[:-1]
            sequential_mask = np.r_[True, sequential_mask, True]
            idx = np.where(sequential_mask)[0]

            cur_plain_groups = []

            for begin, end in zip(idx, idx[1:]):
                if self._check_token_is_plain(classes[begin]):
                    if begin > 0 and self._check_token_is_plain(classes[begin - 1]):
                        cur_plain_groups[-1] += tokens[begin:end].tolist()
                    else:
                        cur_plain_groups.append(tokens[begin:end].tolist())
                    continue
                if end < len(classes) and not self._check_token_is_plain(classes[end]):
                    cur_plain_groups.append([])

                begin_group_token, end_group_token = self._class2range_tokens(classes[begin])

                mid_tokens = np.r_[begin_group_token, tokens[begin:end], end_group_token]
                cur_tokens = np.r_[tokens[:begin], mid_tokens, tokens[end:]]

                tokens_with_classes.append(torch.tensor(cur_tokens, dtype=torch.long))

            plain_groups.append(cur_plain_groups)

        if len(tokens_with_classes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), plain_groups

        tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=self.tokenizer.pad_token_id)
        new_src = tokens_with_classes.to(src.device)

        new_src_mask, new_src_padding_mask = create_mask(self.tokenizer, new_src)

        return new_src, new_src_mask, new_src_padding_mask, plain_groups

    def forward(
            self,
            src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
    ):
        memory, memory_key_padding_mask, plain_groups = self.encode(src, src_mask, src_padding_mask)
        if len(memory) == 0:
            return memory, plain_groups

        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs), plain_groups

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        src, src_mask, src_padding_mask, plain_groups = self._prepare_batch(src, src_mask, src_padding_mask)
        if len(src) == 0:
            return src, src_padding_mask, plain_groups
        return super().encode(src, src_mask, src_padding_mask)[0], src_padding_mask, plain_groups


class Seq2SeqTransformerWithClassEmbeddingsGroupDecoderTrain(Seq2SeqTransformerWithEncoderForClasses):
    """Модель №6 — часть для обучения"""
    name_for_saving = "transformer_with_class_embedding_group_decoder"

    def _check_token_is_plain(self, token):
        return token in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, batch):
        settings = dict(
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False
        )

        batch_spoken_ids = [
            torch.tensor(
                [self.tokenizer.bos_token_id] + elem["spoken_ids"] + [self.tokenizer.eos_token_id],
                dtype=torch.long
            )
            for elem in batch
        ]

        device = next(self.parameters()).device
        batch_spoken_ids = pad_sequence(batch_spoken_ids, padding_value=self.tokenizer.pad_token_id).to(device)
        batch_spoken_mask, batch_spoken_padding_mask = create_mask(self.tokenizer, batch_spoken_ids)

        src_lengths = (batch_spoken_ids != self.tokenizer.pad_token_id).sum(dim=0).cpu()

        all_embeds = super().encode(batch_spoken_ids, batch_spoken_mask, batch_spoken_padding_mask)[0]
        seq_embeds = all_embeds[:, :, :self.emb_size]
        class_embeds = all_embeds[:, :, self.emb_size:]

        embeds_with_classes = []
        targets = []
        src_padding_mask = []
        plain_groups = []

        for elem_i, elem in enumerate(batch):
            classes = np.array([class2ind[cl] for cl in elem["class"]])
            spoken = elem["spoken"][:]  # Копирование данных, чтобы исходные данные не перезатирались
            written = elem["written"][:]

            for i in range(1, len(spoken)):
                spoken[i] = ' ' + spoken[i]
                written[i] = ' ' + written[i]

            sequential_mask = classes[1:] != classes[:-1]
            sequential_mask = np.r_[True, sequential_mask, True]
            idx = np.where(sequential_mask)[0]

            cur_plain_groups = [[]]
            sp_prefix_len = 0

            for begin, end in zip(idx, idx[1:]):
                spoken_ids = self.tokenizer(spoken[begin:end], **settings)["input_ids"]
                spoken_ids = sum(spoken_ids, [])

                sp_prefix_len += len(spoken_ids)

                if self._check_token_is_plain(classes[begin]):
                    if begin == 0 or self._check_token_is_plain(classes[begin - 1]):
                        cur_plain_groups[-1] += spoken_ids
                    else:
                        cur_plain_groups.append(spoken_ids)
                    continue
                if end >= len(classes) or not self._check_token_is_plain(classes[end]):
                    cur_plain_groups.append([])

                written_ids = self.tokenizer(written[begin:end], **settings)["input_ids"]
                written_ids = sum(written_ids, [])

                targets.append(torch.tensor(
                    np.r_[self.tokenizer.bos_token_id, written_ids, self.tokenizer.eos_token_id],
                    dtype=torch.long
                ))

                cur_class_embeds = torch.zeros((class_embeds.shape[0], class_embeds.shape[-1]),
                                               device=class_embeds.device)
                range_idxs = slice(sp_prefix_len - len(spoken_ids), sp_prefix_len)
                cur_class_embeds[range_idxs] = class_embeds[range_idxs, elem_i]

                join_embeds = torch.cat([seq_embeds[:, elem_i], cur_class_embeds], dim=1)

                embeds_with_classes.append(join_embeds)

                src_padding_mask.append(batch_spoken_padding_mask[elem_i])

            plain_groups.append(cur_plain_groups)

        if len(embeds_with_classes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), plain_groups, src_lengths

        embeds_with_classes = torch.stack(embeds_with_classes, dim=1)
        src_padding_mask = torch.stack(src_padding_mask, dim=0)
        targets = pad_sequence(targets, padding_value=self.tokenizer.pad_token_id).to(device)

        return embeds_with_classes, src_padding_mask, targets, plain_groups, src_lengths

    def forward(self, batch):
        memory, memory_key_padding_mask, tgts, plain_groups, src_lengths = self.encode(batch)
        if len(memory) == 0:
            return memory, tgts, plain_groups, src_lengths

        tgt = tgts[:-1]
        tgt_mask, tgt_padding_mask = create_mask(self.tokenizer, tgt=tgt)
        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)

        return self.generator(outs), tgts[1:], plain_groups, src_lengths

    def encode(self, batch):
        return self._prepare_batch(batch)


class Seq2SeqTransformerWithClassEmbeddingsGroupDecoderEval(Seq2SeqTransformerWithEncoderForClasses):
    """Модель №6 — часть для инференса"""
    name_for_saving = "transformer_with_class_embedding_group_decoder"

    def _check_token_is_plain(self, token):
        return token in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def _prepare_batch(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        """Добавляет метку класса для токенов, для которых предсказанный класс не равен PLAIN"""
        memory, src_padding_mask = super().encode(src, src_mask, src_padding_mask)
        class_memory = memory[:, :, self.emb_size:]
        logits = self.transformer_encoder_for_classes.generator(class_memory)  # Get logits from classifier
        batch_classes = logits.argmax(dim=-1).cpu()

        special_tokens_mask = (
                (src == self.tokenizer.bos_token_id) |
                (src == self.tokenizer.eos_token_id) |
                src_padding_mask.T
        )

        batch_classes[special_tokens_mask] = class2ind["PLAIN"]
        batch_classes = batch_classes.T.numpy()

        batch_tokens = src.T.cpu().numpy()
        batch_pad_mask = src_padding_mask.cpu().numpy()
        src_lengths = (src != self.tokenizer.pad_token_id).sum(dim=0).cpu()

        embeds_with_classes = []
        new_src_padding_mask = []  # Создаем новую padding_mask для преобразованного батча
        plain_groups = []
        for batch_i, (classes, tokens, pad_mask) in enumerate(zip(batch_classes, batch_tokens, batch_pad_mask)):
            classes = classes[~pad_mask]
            tokens = tokens[~pad_mask]

            sequential_mask = classes[1:] != classes[:-1]
            sequential_mask = np.r_[True, sequential_mask, True]
            idx = np.where(sequential_mask)[0]

            cur_plain_groups = [[]]

            for begin, end in zip(idx, idx[1:]):
                if self._check_token_is_plain(classes[begin]):
                    if begin == 0 or self._check_token_is_plain(classes[begin - 1]):
                        cur_plain_groups[-1] += tokens[begin:end].tolist()
                    else:
                        cur_plain_groups.append(tokens[begin:end].tolist())
                    continue
                if end >= len(classes) or not self._check_token_is_plain(classes[end]):
                    cur_plain_groups.append([])

                cur_memory = memory[:, batch_i].clone()
                cur_memory[:begin, self.emb_size:] = 0
                cur_memory[end:, self.emb_size:] = 0

                embeds_with_classes.append(cur_memory)
                new_src_padding_mask.append(src_padding_mask[batch_i])

            plain_groups.append(cur_plain_groups)

        if len(embeds_with_classes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), plain_groups

        embeds_with_classes = torch.stack(embeds_with_classes, dim=1)
        new_src_padding_mask = torch.stack(new_src_padding_mask, dim=0)

        return embeds_with_classes, new_src_padding_mask, None, plain_groups, src_lengths

    def forward(
            self,
            src: torch.Tensor, tgt: torch.Tensor,
            src_mask: torch.Tensor, tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
    ):
        memory, memory_key_padding_mask, _, plain_groups, src_lengths = self.encode(src, src_mask, src_padding_mask)
        if len(memory) == 0:
            return memory, _, plain_groups, src_lengths

        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs), _, plain_groups, src_lengths

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
        return self._prepare_batch(src, src_mask, src_padding_mask)


# Второй параметр нужен для совместимости с остальными функциями типа CLI_...
def CLI_model6_train(batch, _, model, tokenizer, device, loss_fn):
    """CLI — Calc(ulate) Loss (for) Iter(ation)"""
    logits, tgt_output = model(batch)[:2]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
    return loss


def mask_for_batch_ids(batch_elem_cnts, eff_batch_elem_cnts):
    eff_batch_ids_len = batch_elem_cnts.sum()
    eff_batch_ids = torch.zeros(eff_batch_ids_len, dtype=torch.long)

    lower_bounds = batch_elem_cnts.cumsum(dim=0)
    lower_bounds = torch.cat((torch.tensor([0]), lower_bounds[:-1]))

    upper_bounds = lower_bounds + eff_batch_elem_cnts

    lower_bounds = lower_bounds[lower_bounds < eff_batch_ids_len]
    upper_bounds = upper_bounds[upper_bounds < eff_batch_ids_len]

    eff_batch_ids.scatter_add_(0, lower_bounds, torch.ones_like(lower_bounds))

    eff_batch_ids.scatter_add_(0, upper_bounds, -torch.ones_like(upper_bounds))

    eff_batch_ids_mask = eff_batch_ids.cumsum(dim=0).bool()
    return eff_batch_ids_mask


def CLI_model5_train_UPC(src, tgt, model, tokenizer, device, loss_fn):
    if len(tgt) == 0:
        return torch.tensor(0., dtype=torch.float, requires_grad=True)

    batch_size = src.shape[1]

    batch_elem_ids = tgt[0].cpu()
    tgt = tgt[1:]

    tgt_input = tgt[:-1, :]
    src_mask, src_padding_mask, tgt_mask, tgt_padding_mask = create_mask(tokenizer, src, tgt_input)

    memory, memory_key_padding_mask, plain_groups = model.encode(src, src_mask, src_padding_mask)

    # Выравниваем группы у src и tgt
    src_batch_elem_cnts = torch.tensor([len(group) - 1 for group in plain_groups])
    tgt_batch_elem_cnts = torch.bincount(batch_elem_ids, minlength=batch_size)
    eff_batch_elem_cnts = torch.min(src_batch_elem_cnts, tgt_batch_elem_cnts)

    src_eff_batch_ids_mask = mask_for_batch_ids(src_batch_elem_cnts, eff_batch_elem_cnts)
    tgt_eff_batch_ids_mask = mask_for_batch_ids(tgt_batch_elem_cnts, eff_batch_elem_cnts)

    # Repair everything
    memory = memory[:, src_eff_batch_ids_mask]
    memory_key_padding_mask = memory_key_padding_mask[src_eff_batch_ids_mask]
    tgt = tgt[:, tgt_eff_batch_ids_mask]
    tgt_input = tgt[:-1, :]
    tgt_padding_mask = tgt_padding_mask[tgt_eff_batch_ids_mask]

    # Decode and calc loss
    out_embeds = model.decode(tgt_input, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
    logits = model.generator(out_embeds)

    tgt_output = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

    return loss


def CLI_model4_train_UPC(src, tgt, model, tokenizer, device, loss_fn):
    """Alias for model #4."""
    return CLI_model5_train_UPC(src, tgt, model, tokenizer, device, loss_fn)


def repair_batch_lengths(memory, src_lengths, plain_groups, max_additional_tokens):
    batch_size = memory.shape[1]  # src.shape[1]

    # Восстанавливаем максимальные длины строк в батче по максимальным длинам строк из src
    # (у нас один объект из src распадается на несколько объектов в батче).
    group_lens = [len(group) - 1 for group in plain_groups]
    idxs = np.cumsum(group_lens)
    idxs = idxs[idxs < batch_size]

    batch_lengths = np.zeros(batch_size, dtype=int)
    batch_lengths[idxs] = 1
    batch_lengths = np.cumsum(batch_lengths)
    batch_lengths = src_lengths[batch_lengths]

    batch_lengths += max_additional_tokens
    # Закончили восстановление максимальных длин строк в батче

    return batch_lengths


@torch.inference_mode()
def _memory_greedy_decode(model, memory, memory_key_padding_mask, plain_groups, src_lengths, max_additional_tokens, start_symbol, tokenizer, device):
    if len(memory) == 0:
        return memory, plain_groups

    batch_lengths = repair_batch_lengths(memory, src_lengths, plain_groups, max_additional_tokens)

    result = memory_greedy_decode(
        model, memory, memory_key_padding_mask,
        batch_lengths, start_symbol, tokenizer, device
    )

    return result, plain_groups


@torch.inference_mode()
def _memory_beam_search(
        model,
        memory, memory_key_padding_mask, plain_groups,
        src_lengths, max_additional_tokens,
        start_symbol, tokenizer, device,
        beam_width=10
):
    if len(memory) == 0:
        return memory, plain_groups

    batch_lengths = repair_batch_lengths(memory, src_lengths, plain_groups, max_additional_tokens)

    result = memory_beam_search(
        model, memory, memory_key_padding_mask,
        batch_lengths, start_symbol, tokenizer, device, beam_width
    )

    return result, plain_groups


def _repair_translation_with_plain_groups(translations, plain_groups, tokenizer):
    translations = translations.T.numpy()
    translations_idx = 0

    repaired_translations = []
    for group in plain_groups:
        result = group[0]
        for chunk in group[1:]:
            result += translations[translations_idx].tolist()
            result += chunk
            translations_idx += 1
        repaired_translations.append(result)

    result = decode(tokenizer, repaired_translations, batch_first=True)
    return result


@torch.inference_mode()
def seq2seq_transformer_classes_groups_decode(
        model, src, max_additional_tokens,
        start_symbol, tokenizer, device, beam_width
):
    model_enc, src_lengths = model_encode(model, src, tokenizer, device, return_src_lengths=True)
    memory, memory_key_padding_mask, plain_groups = model_enc

    if beam_width is None:
        return _memory_greedy_decode(
            model, memory, memory_key_padding_mask, plain_groups, src_lengths,
            max_additional_tokens, start_symbol, tokenizer, device
        )

    return _memory_beam_search(
        model, memory, memory_key_padding_mask, plain_groups, src_lengths,
        max_additional_tokens, start_symbol, tokenizer, device, beam_width
    )


@torch.inference_mode()
@make_fp_function
def seq2seq_transformer_classes_groups_translate(
        model, tokenizer, device, srcs,
        max_additional_tokens=5, beam_width=None
):
    model = model.to(device)
    model.eval()

    src = prepare_srcs_for_translate(srcs, tokenizer)

    translations, plain_groups = seq2seq_transformer_classes_groups_decode(
        model, src, max_additional_tokens, tokenizer.bos_token_id, tokenizer, device, beam_width
    )

    return _repair_translation_with_plain_groups(translations, plain_groups, tokenizer)


@torch.inference_mode()
def model6_decode(model, batch, max_additional_tokens, start_symbol, tokenizer, device, beam_width):
    model = model.to(device)
    model.eval()

    if isinstance(batch, torch.Tensor):
        # Inference mode
        src = batch.to(device)
        src_mask, src_padding_mask = create_mask(tokenizer, src)
        encode_args = src, src_mask, src_padding_mask
    else:
        # Train mode
        encode_args = (batch, )

    memory, memory_key_padding_mask, tgts, plain_groups, src_lengths = model.encode(*encode_args)

    if beam_width is None:
        return _memory_greedy_decode(model, memory, memory_key_padding_mask, plain_groups, src_lengths,
                                     max_additional_tokens, start_symbol, tokenizer, device)

    return _memory_beam_search(model, memory, memory_key_padding_mask, plain_groups, src_lengths,
                               max_additional_tokens, start_symbol, tokenizer, device, beam_width)


@torch.inference_mode()
@make_fp_function
def model6_translate(model, tokenizer, device, srcs, max_additional_tokens=5, beam_width=None):
    model = model.to(device)
    model.eval()

    translations, plain_groups = model6_decode(
        model, srcs, max_additional_tokens,
        tokenizer.bos_token_id, tokenizer, device, beam_width
    )

    return _repair_translation_with_plain_groups(translations, plain_groups, tokenizer)
