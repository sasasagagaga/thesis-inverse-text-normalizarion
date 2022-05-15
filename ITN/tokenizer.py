
# *** Токенизатор (tokenizer) *** #

# Нужен для того, чтобы подготовить последовательность токенов для подачи в Transformer-подобные модели.

# Common imports
import os
import itertools

# Tokenizer for Transformer-like models
from transformers import RobertaTokenizerFast, AutoTokenizer

# Local imports
from .core import data_paths
from .data.core import class2ind

DEFAULT_TOKENIZER_NAME = "my_ru"
POSSIBLE_TOKENIZER_NAMES = (
    "my_ru",
    "rubert",
    "xlm",
    "helsinki-nlp-ru"
)

special_tokens = {
    "[]": {
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]'
    },
    "<>": {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'unk_token': '<unk>',
        'sep_token': '<sep>',
        'pad_token': '<pad>',
        'cls_token': '<cls>',
        'mask_token': '<mask>'
    }
}


def load_tokenizer(tokenizer_name=None):
    tokenizer_name = tokenizer_name or DEFAULT_TOKENIZER_NAME

    if tokenizer_name not in POSSIBLE_TOKENIZER_NAMES:
        raise ValueError(f"Tokenizer name should be from {POSSIBLE_TOKENIZER_NAMES}")

    if tokenizer_name.lower() == "my_ru":
        # *** My RU tokenizer *** #
        language = "ru"
        base_path = os.path.join(data_paths[language], "tokenizers", "my_ru")
        my_ru_tokenizer = RobertaTokenizerFast(
            os.path.join(base_path, "vocab.json"),
            os.path.join(base_path, "merges.txt"),
            os.path.join(base_path, "tokenizer.json"),
            **special_tokens["<>"]
        )
        my_ru_tokenizer.add_special_tokens(special_tokens["<>"])
        return my_ru_tokenizer
    elif tokenizer_name == "rubert":
        # *** RuBERT tokenizer *** #
        rubert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", **special_tokens["[]"])
        rubert_tokenizer.add_special_tokens(special_tokens["[]"])
        return rubert_tokenizer
    elif tokenizer_name.lower() == "xlm":
        # *** XLM tokenizer *** #
        xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", **special_tokens["<>"])
        xlm_tokenizer.add_special_tokens(special_tokens["<>"])
        return xlm_tokenizer
    elif tokenizer_name.lower() == "helsinki-nlp-ru":
        # *** Helsinki-NLP RU tokenizer *** #
        ru_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en", **special_tokens["<>"])
        ru_tokenizer.add_special_tokens(special_tokens["<>"])
        return ru_tokenizer


def add_class_tokens(tokenizer):
    tokens_to_add = [f"[{x}]" for x in class2ind.keys()]
    tokenizer.add_tokens(tokens_to_add)


def add_range_class_tokens(tokenizer):
    tokens_to_add = list(itertools.chain.from_iterable(
        (f"[{x}]", f"[/{x}]") for x in class2ind.keys()
    ))
    tokenizer.add_tokens(tokens_to_add)


def get_tokenizer(tokenizer_name=None, add_class_tokens_fl=False, add_range_class_tokens_fl=False):
    tokenizer = load_tokenizer(tokenizer_name)

    if add_class_tokens_fl:
        add_class_tokens(tokenizer)
    if add_range_class_tokens_fl:
        add_range_class_tokens(tokenizer)

    return tokenizer


def decode(tokenizer, sequences, batch_first=False):
    """batch_first is True, если первая размерность у sequences отвечает за размер батча"""
    if not batch_first:
        sequences = sequences.T
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
