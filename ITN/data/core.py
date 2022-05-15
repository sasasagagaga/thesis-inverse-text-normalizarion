
POSSIBLE_DATA_TARGETS = ("seq2seq", "classes")


class2ind = {
    'PLAIN': 0,
    'PUNCT': 1,
    'CARDINAL': 2,
    'VERBATIM': 3,
    'LETTERS': 4,
    'DATE': 5,
    'MEASURE': 6,
    'TELEPHONE': 7,
    'ORDINAL': 8,
    'DIGIT': 9,
    'ELECTRONIC': 10,
    'MONEY': 11,
    'TIME': 12,
    'FRACTION': 13,
    'DECIMAL': 14
}

ind2class = {
    0: 'PLAIN',
    1: 'PUNCT',
    2: 'CARDINAL',
    3: 'VERBATIM',
    4: 'LETTERS',
    5: 'DATE',
    6: 'MEASURE',
    7: 'TELEPHONE',
    8: 'ORDINAL',
    9: 'DIGIT',
    10: 'ELECTRONIC',
    11: 'MONEY',
    12: 'TIME',
    13: 'FRACTION',
    14: 'DECIMAL'
}

CLASS_PAD_IDX = 999
