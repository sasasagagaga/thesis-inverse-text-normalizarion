# TODO: Допилить этот скрипт с предобработкой данных

from functools import partial

import numpy as np

from ..utils import make_fp_function
from ..tokenizer import load_tokenizer



# ' '.join(dataset_val[0]['spoken'])[9:14]

# ' '.join(dataset_val[0]["spoken"])

# np.cumsum([0] + [len(x) + 1 for i, x in enumerate(dataset_val[0]["spoken"])])

# (
#     ' '.join(dataset_val[0]['spoken'])[0:3],
#     ' '.join(dataset_val[0]['spoken'])[4:14],
#     ' '.join(dataset_val[0]['spoken'])[15:16],
#     ' '.join(dataset_val[0]['spoken'])[17:71],
#     ' '.join(dataset_val[0]['spoken'])[72:73],
#     ' '.join(dataset_val[0]['spoken'])[74:75],
# )

# [x[0] for x in default_tokenizer(' '.join(dataset_val[0]['spoken']), return_offsets_mapping=True)["offset_mapping"]]


# default_tokenizer(' '.join(dataset_val[0]['spoken']), return_offsets_mapping=True)

# default_tokenizer.convert_ids_to_tokens(
#     default_tokenizer(' '.join(dataset_val[0]['spoken']), return_offsets_mapping=True)["input_ids"]
# )

# default_tokenizer(dataset_val[0]['spoken'], is_split_into_words=True, return_offsets_mapping=True)

# default_tokenizer.convert_ids_to_tokens(
#     default_tokenizer(dataset_val[0]['spoken'], is_split_into_words=True)["input_ids"]
# )

# for x in [[186, 11944], [226, 3505, 6053], [14],
#           [210, 1055, 1155, 1918, 1765, 1166, 1170, 2193, 1121], [15], [18]]:
#     print(default_tokenizer.convert_ids_to_tokens(x))

# default_tokenizer.convert_tokens_to_ids("фантастики")

# dataset_val[0]

# default_tokenizer.decode([5562, 7118, 6053, 1075, 1276, 1918, 1765, 1166, 1170, 2193, 1121, 1073, 1012])

# default_tokenizer.decode(
#     [1332, 1276, 1918, 1733, 1166, 1170, 1977, 1121, 1012, 1503,
#      1142, 1502, 2034, 1820, 1166, 1170, 1977, 1121, 1012, 1563,
#      4383, 2097, 227, 1136, 4720, 4452, 1249, 12753, 1099, 2799,
#      1560, 225, 1036, 1134, 220, 1759, 1559, 1206, 1012]
# )

# tmpb

# tmpa = default_tokenizer(dataset_train[44323:44324]["written"], is_split_into_words=True, return_offsets_mapping=True)

# tmp = tmpa["offset_mapping"]

# tmp

# tmpb = [np.cumsum([y[0] == 0 for y in x]) - 1 for x in tmp]

# # default_tokenizer.convert_ids_to_tokens(
# #     default_tokenizer(dataset_val[0]["spoken"], is_split_into_words=True, return_offsets_mapping=True)["input_ids"]
# # )

# [np.array(y)[x].tolist() for x, y in zip(tmpb, dataset_train[44323:44324]["class"])]

# # tmpa

# tmpa["input_ids"]

# tmpa["offset_mapping"][0]

# default_tokenizer.convert_tokens_to_ids('▁')

# tmpa["input_ids"][0]

# banned_sentences = {55575}

# dataset_train.filter(lambda x: x["sentence_id"] not in banned_sentences)


# dataset_train[44323]


# list(zip(default_tokenizer.convert_ids_to_tokens(tmpa["input_ids"][0]), tmpa["offset_mapping"][0]))

# tmpa

# # dataset_val[88]

# tmp

# tmpb


# dataset_train[44323]

# [
#     np.cumsum([
#         idx1 == 0 and token != default_tokenizer.convert_tokens_to_ids('▁')
#         for (idx1, idx2), token
#         in zip(idxs, text_tokens)
#     ]) - 1
#     for idxs, text_tokens
#     in zip(
#         [[(0, 6), (0, 1), (0, 1), (0, 1), (0, 5), (5, 6), (6, 9), (0, 4), (0, 1), (0, 2), (2, 6), (6, 11), (0, 4), (4, 5), (5, 8), (0, 4), (0, 1), (0, 2), (2, 4), (4, 7), (0, 2), (0, 1), (1, 4), (4, 6), (0, 1)]],
#         [[4167, 1081, 429, 24, 10505, 178, 1627, 3412, 1157, 2744, 1758, 3364, 1121, 192, 4391, 3412, 1157, 2183, 13403, 1278, 1154, 1015, 1461, 1028, 2987]]
#     )
# ]


# dataset_train[44323:44324]["class"]

# mapped_classes = [
#     np.array(classes)[idxs].tolist()
#     for idxs, classes
#     in zip(indices_for_classes, examples["class"])
# ]

# seq2seq_encoder(dataset_train[44323:44324]);


def encode_for_seq2seq(examples, tokenizer=None):
    assert tokenizer is not None, "You should provide tokenizer"

    #     join_texts = lambda texts: [' '.join(x) for x in texts]
    #     joined_spoken, joined_written = map(join_texts, [examples["spoken"], examples["written"]])
    #     joined_spoken = [' '.join(x) for x in examples["spoken"]]
    #     joined_written = [' '.join(x) for x in examples["written"]]

    settings = {}
    if tokenizer.name_or_path == "xlm-roberta-base":
        settings = dict(
            truncation=True,
            padding="max_length"
        )
    elif tokenizer.name_or_path == "DeepPavlov/rubert-base-cased":
        settings = dict(
            #             **special_tokens_for_tokenization,  # Fails
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False
        )
    else:
        settings = dict(
            #             **special_tokens_for_tokenization,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            is_split_into_words=True,
            return_offsets_mapping=True
        )

    # Добавляю BOS и EOS явно при генерации батчей, поэтому тут не надо.
    #     tokenize = lambda joined_texts: [
    #         [tokenizer.bos_token_id] + x + [tokenizer.eos_token_id]
    #         for x
    #         in tokenizer(joined_texts, **settings)["input_ids"]
    #     ]
    tokenize = lambda texts: tokenizer(texts, **settings)  # ["input_ids"]

    tokenized_spoken, tokenized_written = map(tokenize, [examples["spoken"], examples["written"]])

    def map_classes(tokenized_texts):
        offset_mapping = tokenized_texts["offset_mapping"]
        texts_tokens = tokenized_texts["input_ids"]
        #         print(offset_mapping)
        #         print(texts_tokens)
        #         print(list(zip(offset_mapping, texts_tokens)))
        indices_for_classes = [
            np.cumsum([
                idx1 == 0 and token != tokenizer.convert_tokens_to_ids('▁')
                for (idx1, idx2), token
                in zip(idxs, text_tokens)
            ]) - 1
            for idxs, text_tokens
            in zip(offset_mapping, texts_tokens)
        ]
        mapped_classes = [
            np.array(classes)[np.clip(idxs, 0, len(classes) - 1)].tolist()
            for idxs, classes
            in zip(indices_for_classes, examples["class"])
        ]
        return mapped_classes

    classes_spoken, classes_written = map(map_classes, [tokenized_spoken, tokenized_written])

    return {
        "spoken_ids": tokenized_spoken["input_ids"],
        "spoken_classes": classes_spoken,
        #         "spoken_attention_mask": tokenized_spoken["attention_mask"],
        "written_ids": tokenized_written["input_ids"],
        "written_classes": classes_written,
        #         "written_attention_mask": tokenized_written["attention_mask"],
    }


@make_fp_function
def filter_for_seq2seq(examples, max_length):
    # =None
    # assert max_length is not None, "You should provide max_length"

    ids_spoken, ids_written = examples["spoken_ids"], examples["written_ids"]

    return max(len(ids_spoken), len(ids_written)) < max_length


#     ids_filtered = [
#         max(len(x), len(y)) < max_sentence_length
#         for x, y
#         in zip(ids_spoken, ids_written)
#     ]

#     return ids_filtered

# %%


default_tokenizer = load_tokenizer("my_ru")

max_sentence_length = 126

seq2seq_encoder = partial(
    encode_for_seq2seq,
    tokenizer=default_tokenizer
)

seq2seq_filter = filter_for_seq2seq(max_length=max_sentence_length)
# partial(
#     filter_for_seq2seq,
#     max_sentence_length=max_sentence_length
# )

# seq2seq_encoder(dataset_train[:1])["written_ids"]
# rubert_tokenizer.convert_ids_to_tokens(seq2seq_encoder(dataset_train[:1])["written_ids"][0])

# %%

# for i in tqdm(range(0, len(dataset_train), 1000)):
#     examples = dataset_train[i:i+1000]
#     try:
#         seq2seq_encoder(examples)
#     except IndexError:
#         print(i, i + 1000)

# for i in tqdm(range(0, len(dataset_val), 250)):
#     examples = dataset_train[i:i+1000]
#     try:
#         seq2seq_encoder(examples)
#     except IndexError:
#         print(i, i + 1000)

# %%

# # % % time
#
# dataset_train = dataset_train.map(seq2seq_encoder, batched=True).filter(seq2seq_filter)
# dataset_train.save_to_disk(f"{data_paths['ru']}/dataset_train_{filename_dataset_id}_tokenized_for_classes")
#
# # %%
#
# # % % time
#
# dataset_val = dataset_val.map(seq2seq_encoder, batched=True).filter(seq2seq_filter)
# dataset_val.save_to_disk(f"{data_paths['ru']}/dataset_val_{filename_dataset_id}_tokenized_for_classes")
#
# # %%
#
# dataset_train = datasets.load_from_disk(f"{data_paths['ru']}/dataset_train_{filename_dataset_id}_tokenized_for_classes")
# dataset_val = datasets.load_from_disk(f"{data_paths['ru']}/dataset_val_{filename_dataset_id}_tokenized_for_classes")
