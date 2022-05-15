from functools import wraps, partial
import os
import random
import re

import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Everything was seeded with {seed}.")


def get_device(try_device="cuda"):
    try_device = torch.device(try_device)

    if try_device.type == "cpu":
        device = try_device
    else:
        # setting device on GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")

    return device


# Декоратор, делает из функции функцию в стиле функционального программирования, то есть если у функции
# несколько аргументов, то вызов функции только от части аргументов — это тоже функция.
def make_fp_function(func):  # fp stands for Functional Programming
    @wraps(func)
    def wrapper(*args, call_function=True, **kwargs):
        new_func = func
        if len(args) + len(kwargs) > 0:
            # Если есть аргументы для функции, тогда добавляем их к функции и
            # затем новую функцию обмазываем заново нашим декоратором.
            new_func = make_fp_function(partial(new_func, *args, **kwargs))

        if call_function:
            try:
                # Пробуем вызвать функцию. Если вызвалась, значит,
                # все необходимые позиционные аргументы были в нее переданы.
                return new_func()
            except TypeError as e:
                # Если не вызвалась, тогда проверяем, почему.
                # Если из-за ошибки, что не хватает некоторых аргументов, тогда возвращаем текущую полученную функцию,
                # иначе поднимаем возникшее исключение.
                if not re.search(r"missing (\d)+ required positional argument", str(e)):
                    # Здесь будут ошибки типа "got multiple values for argument" и т.д.
                    raise
        return wrapper
    return wrapper
