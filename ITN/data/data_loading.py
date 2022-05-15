
# Common libraries
import os
from functools import partial

import numpy as np

# Pytorch
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Simple and fast dataset for texts
import datasets

# Local imports
from ..core import data_paths
from ..tokenizer import get_tokenizer
from .core import class2ind, ind2class, CLASS_PAD_IDX, POSSIBLE_DATA_TARGETS


def generate_batch(data_batch, tokenizer):
    sp_batch, wr_batch = [], []

    data_batch = map(lambda x: (x["spoken_ids"], x["written_ids"]), data_batch)
    for sp_item, wr_item in data_batch:
        sp_batch.append(torch.tensor(
            [tokenizer.bos_token_id] + sp_item + [tokenizer.eos_token_id]
        ))
        wr_batch.append(torch.tensor(
            [tokenizer.bos_token_id] + wr_item + [tokenizer.eos_token_id]
        ))

    sp_batch = pad_sequence(sp_batch, padding_value=tokenizer.pad_token_id)
    wr_batch = pad_sequence(wr_batch, padding_value=tokenizer.pad_token_id)

    return sp_batch, wr_batch


def generate_batch_for_classes(data_batch, tokenizer):
    id_batch, cl_batch = [], []

    data_batch = map(lambda x: (x["spoken_ids"], x["spoken_classes"]), data_batch)
    for id_item, cl_item in data_batch:
        id_batch.append(torch.tensor(id_item))
        cl_batch.append(torch.tensor([class2ind[x] for x in cl_item]))

    id_batch = pad_sequence(id_batch, padding_value=tokenizer.pad_token_id)
    cl_batch = pad_sequence(cl_batch, padding_value=CLASS_PAD_IDX)

    return id_batch, cl_batch


def generate_batch_for_classes_labeling(batch, tokenizer):
    """batch from dataset with default collate_fn"""
    settings = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False
    )

    def _class2token(cur_class):
        token_for_group = ind2class[cur_class]
        token_for_group = f"[{token_for_group}]"
        token_for_group = tokenizer(token_for_group)["input_ids"]
        if len(token_for_group) > 1:
            raise ValueError("Strange token for class (after tokenization has more then 1 token)")
        return token_for_group[0]

    def _check_token_is_plain(token_class):
        return token_class in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def pytorchify(tokens):
        return torch.LongTensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id])

    tokens_with_classes = []
    targets = []
    for elem in batch:
        classes = np.array([class2ind[cl] for cl in elem["class"]])
        spoken = elem["spoken"][:]
        all_spoken_ids = elem["spoken_ids"]
        cur_targets = elem["written_ids"]

        for i in range(1, len(spoken)):
            spoken[i] = ' ' + spoken[i]

        sequential_mask = classes[1:] != classes[:-1]
        sequential_mask = np.r_[True, sequential_mask, True]
        idx = np.where(sequential_mask)[0]

        sp_prefix_len = 0
        new_tokens = []
        for begin, end in zip(idx, idx[1:]):
            spoken_ids = tokenizer(spoken[begin:end], **settings)["input_ids"]
            spoken_ids = sum(spoken_ids, [])

            sp_prefix_len += len(spoken_ids)

            cls = classes[begin]
            if _check_token_is_plain(cls):
                new_tokens += all_spoken_ids[sp_prefix_len - len(spoken_ids):sp_prefix_len]  # spoken_ids???
            else:
                class_token = _class2token(cls)
                mid_tokens = np.empty(2 * len(spoken_ids), dtype=int)
                mid_tokens[0::2] = class_token
                mid_tokens[1::2] = all_spoken_ids[sp_prefix_len - len(spoken_ids):sp_prefix_len]
                new_tokens += mid_tokens.tolist()

        tokens_with_classes.append(pytorchify(new_tokens))
        targets.append(pytorchify(cur_targets))

    tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id)

    return tokens_with_classes, targets


def generate_batch_for_classes_range_labeling(batch, tokenizer):
    """batch from dataset with default collate_fn"""
    settings = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False
    )

    def _class2range_tokens(cur_class):
        cur_class = ind2class[cur_class]
        tokens_for_group = [f"[{cur_class}]", f"[/{cur_class}]"]
        begin_group_token, end_group_token = tokenizer(tokens_for_group)["input_ids"]
        if len(begin_group_token) != 1:
            raise ValueError("Strange begin group token for class (after tokenization has not 1 token)")
        if len(end_group_token) != 1:
            raise ValueError("Strange end group token for class (after tokenization has not 1 token)")
        return begin_group_token[0], end_group_token[0]

    def _check_token_is_plain(token_class):
        return token_class in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    def pytorchify(tokens):
        return torch.LongTensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id])

    tokens_with_classes = []
    targets = []
    for elem in batch:
        classes = np.array([class2ind[cl] for cl in elem["class"]])
        spoken = elem["spoken"][:]
        all_spoken_ids = elem["spoken_ids"]
        cur_targets = elem["written_ids"]

        for i in range(1, len(spoken)):
            spoken[i] = ' ' + spoken[i]

        sequential_mask = classes[1:] != classes[:-1]
        sequential_mask = np.r_[True, sequential_mask, True]
        idx = np.where(sequential_mask)[0]

        sp_prefix_len = 0
        new_tokens = []
        for begin, end in zip(idx, idx[1:]):
            spoken_ids = tokenizer(spoken[begin:end], **settings)["input_ids"]
            spoken_ids = sum(spoken_ids, [])

            sp_prefix_len += len(spoken_ids)

            cur_tokens = all_spoken_ids[sp_prefix_len - len(spoken_ids):sp_prefix_len]
            cls = classes[begin]
            if not _check_token_is_plain(cls):
                begin_group_token, end_group_token = _class2range_tokens(cls)
                cur_tokens = [begin_group_token] + cur_tokens + [end_group_token]
            new_tokens += cur_tokens

        tokens_with_classes.append(pytorchify(new_tokens))
        targets.append(pytorchify(cur_targets))

    tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id)

    return tokens_with_classes, targets


def generate_batch_for_classes_group_decoding(batch, tokenizer):
    """batch from dataset with default collate_fn"""
    settings = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False
    )

    def _class2token(cur_class):
        token_for_group = ind2class[cur_class]
        token_for_group = f"[{token_for_group}]"
        token_for_group = tokenizer(token_for_group)["input_ids"]
        if len(token_for_group) > 1:
            raise ValueError("Strange token for class (after tokenization has more then 1 token)")
        return token_for_group[0]

    tokens_with_classes = []
    targets = []
    for elem in batch:
        classes = np.array([class2ind[cl] for cl in elem["class"]])
        spoken = elem["spoken"][:]
        written = elem["written"][:]
        all_spoken_ids = elem["spoken_ids"]

        for i in range(1, len(spoken)):
            spoken[i] = ' ' + spoken[i]
            written[i] = ' ' + written[i]

        sequential_mask = classes[1:] != classes[:-1]
        sequential_mask = np.r_[True, sequential_mask, True]
        idx = np.where(sequential_mask)[0]

        sp_prefix_len, wr_prefix_len = 0, 0

        for begin, end in zip(idx, idx[1:]):
            spoken_ids = tokenizer(spoken[begin:end], **settings)["input_ids"]
            spoken_ids = sum(spoken_ids, [])

            sp_prefix_len += len(spoken_ids)

            if classes[begin] == class2ind["PLAIN"]:
                continue

            written_ids = tokenizer(written[begin:end], **settings)["input_ids"]
            written_ids = sum(written_ids, [])
            wr_prefix_len += len(written_ids)

            targets.append(torch.tensor(
                np.r_[tokenizer.bos_token_id, written_ids, tokenizer.eos_token_id],
                dtype=torch.long
            ))

            class_token = _class2token(classes[begin])
            mid_tokens = np.empty(2 * len(spoken_ids), dtype=int)
            mid_tokens[0::2] = class_token
            mid_tokens[1::2] = all_spoken_ids[sp_prefix_len - len(spoken_ids):sp_prefix_len]

            cur_tokens = np.r_[
                tokenizer.bos_token_id,
                all_spoken_ids[:sp_prefix_len - len(spoken_ids)],
                mid_tokens,
                all_spoken_ids[sp_prefix_len:],
                tokenizer.eos_token_id
            ]

            tokens_with_classes.append(torch.tensor(cur_tokens, dtype=torch.long))

    tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id)

    return tokens_with_classes, targets


def generate_batch_for_classes_range_group_decoding(batch, tokenizer):
    """Batch collator for model №5
    `batch` is from dataset with default collate_fn"""
    settings = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False
    )

    def _class2range_tokens(cur_class):
        cur_class = ind2class[cur_class]
        tokens_for_group = [f"[{cur_class}]", f"[/{cur_class}]"]
        begin_group_token, end_group_token = tokenizer(tokens_for_group)["input_ids"]
        if len(begin_group_token) != 1:
            raise ValueError("Strange begin group token for class (after tokenization has not 1 token)")
        if len(end_group_token) != 1:
            raise ValueError("Strange end group token for class (after tokenization has not 1 token)")
        return begin_group_token[0], end_group_token[0]

    tokens_with_classes = []
    targets = []
    for elem in batch:
        classes = np.array([class2ind[cl] for cl in elem["class"]])
        spoken = elem["spoken"][:]
        written = elem["written"][:]
        all_spoken_ids = elem["spoken_ids"]

        for i in range(1, len(spoken)):
            spoken[i] = ' ' + spoken[i]
            written[i] = ' ' + written[i]

        sequential_mask = classes[1:] != classes[:-1]
        sequential_mask = np.r_[True, sequential_mask, True]
        idx = np.where(sequential_mask)[0]

        sp_prefix_len, wr_prefix_len = 0, 0

        for begin, end in zip(idx, idx[1:]):
            spoken_ids = tokenizer(spoken[begin:end], **settings)["input_ids"]
            spoken_ids = sum(spoken_ids, [])

            sp_prefix_len += len(spoken_ids)

            if classes[begin] == class2ind["PLAIN"]:
                continue

            written_ids = tokenizer(written[begin:end], **settings)["input_ids"]
            written_ids = sum(written_ids, [])
            wr_prefix_len += len(written_ids)

            targets.append(torch.tensor(
                np.r_[tokenizer.bos_token_id, written_ids, tokenizer.eos_token_id],
                dtype=torch.long
            ))

            begin_group_token, end_group_token = _class2range_tokens(classes[begin])
            mid_tokens = np.r_[
                begin_group_token,
                all_spoken_ids[sp_prefix_len - len(spoken_ids):sp_prefix_len],
                end_group_token
            ]

            cur_tokens = np.r_[
                tokenizer.bos_token_id,
                all_spoken_ids[:sp_prefix_len - len(spoken_ids)],
                mid_tokens,
                all_spoken_ids[sp_prefix_len:],
                tokenizer.eos_token_id
            ]

            tokens_with_classes.append(torch.tensor(cur_tokens, dtype=torch.long))

    tokens_with_classes = pad_sequence(tokens_with_classes, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id)

    return tokens_with_classes, targets


def get_datasets(tokenizer_name, language, data_target):  # =DEFAULT_TOKENIZER_NAME, ="ru"
    data_target = data_target.lower()
    assert data_target in POSSIBLE_DATA_TARGETS, f"data_target should be from {POSSIBLE_DATA_TARGETS}"

    datasets_path = os.path.join(data_paths[language], f"{data_target}_tokenized", tokenizer_name)

    return tuple(datasets.load_from_disk(os.path.join(datasets_path, mode)) for mode in ("train", "val"))


def get_dataloaders(tokenizer_name, language, data_target, generate_batch_function, batch_size: int, **gen_kwargs):
    dataset_train, dataset_val = get_datasets(tokenizer_name, language, data_target)

    if "tokenizer" in gen_kwargs:
        gen_kwargs["tokenizer"] = gen_kwargs["tokenizer"] or get_tokenizer(tokenizer_name)
    if isinstance(generate_batch_function, str):
        if generate_batch_function == "seq2seq":
            generate_batch_function = generate_batch
        elif generate_batch_function == "classes":
            generate_batch_function = generate_batch_for_classes
    collate_fn = partial(generate_batch_function, **gen_kwargs)

    return (
        DataLoader(dataset_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True),
        DataLoader(dataset_val, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    )


def generate_batch_for_classes_range_group_decoding_UPC(batch, tokenizer):
    """
    Batch collator for model №4 and 5
    `batch` is from dataset with default collate_fn.
    UPC — Using Predicted Classes (during train).
    """
    def _check_token_is_plain(token_class):
        return token_class in (class2ind["PLAIN"], class2ind["PUNCT"], class2ind["VERBATIM"], class2ind["MEASURE"])

    settings = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False
    )

    sources = []
    targets = []
    targets_elem_ids = []  # Чтобы знать, для какого элемента батча этот таргет
    for elem_i, elem in enumerate(batch):
        classes = np.array([class2ind[cl] for cl in elem["class"]])
        spoken = elem["spoken"][:]
        written = elem["written"][:]
        all_spoken_ids = elem["spoken_ids"]

        for i in range(1, len(spoken)):
            spoken[i] = ' ' + spoken[i]
            written[i] = ' ' + written[i]

        sequential_mask = classes[1:] != classes[:-1]
        sequential_mask = np.r_[True, sequential_mask, True]
        idx = np.where(sequential_mask)[0]

        sp_prefix_len, wr_prefix_len = 0, 0

        for begin, end in zip(idx, idx[1:]):
            spoken_ids = tokenizer(spoken[begin:end], **settings)["input_ids"]
            spoken_ids = sum(spoken_ids, [])

            sp_prefix_len += len(spoken_ids)

            if _check_token_is_plain(classes[begin]):
                continue

            written_ids = tokenizer(written[begin:end], **settings)["input_ids"]
            written_ids = sum(written_ids, [])
            wr_prefix_len += len(written_ids)

            targets.append(torch.tensor(
                np.r_[tokenizer.bos_token_id, written_ids, tokenizer.eos_token_id],
                dtype=torch.long
            ))
            targets_elem_ids.append(elem_i)

        sources.append(torch.tensor(
            [tokenizer.bos_token_id] + all_spoken_ids + [tokenizer.eos_token_id],
            dtype=torch.long
        ))

    sources = pad_sequence(sources, padding_value=tokenizer.pad_token_id)
    if len(targets) == 0:
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id)
    targets_elem_ids = torch.tensor([targets_elem_ids], dtype=torch.long)

    targets = torch.cat([targets_elem_ids, targets])

    return sources, targets


def generate_batch_for_classes_group_decoding_UPC(batch, tokenizer):
    """
    Alias for model #4.

    Batch collator for model №4 and 5
    `batch` is from dataset with default collate_fn.
    UPC — Using Predicted Classes (during train).
    """
    return generate_batch_for_classes_range_group_decoding_UPC(batch, tokenizer)
