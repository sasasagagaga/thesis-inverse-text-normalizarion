# *** UTILITIES *** #

import argparse

# Model parameters configurator

# Configurable parameters
emb_size_vars = [128, 256, 512, 1024]
num_encoder_layers_vars = [2, 4, 6]
lr_vars = [3e-4, 1e-3]

# Functionally configurable parameters
num_decoder_layers_vars = [
    lambda num_encoder_layers: num_encoder_layers,
    lambda num_encoder_layers: num_encoder_layers + 1
]

# Chosen parameters
dim_feedforward_vars = [
    lambda emb_size: 1 * emb_size,  # NOTE: The best variant
    lambda emb_size: 2 * emb_size,
    lambda emb_size: 4 * emb_size
]

nhead_vars = [
    lambda emb_size: emb_size // 64
]


# Configurator
def configure_model_params(
        *,
        emb_size=512, num_encoder_layers=3, lr=3e-4,
        nhead_var=nhead_vars[0],
        num_decoder_layers_var=num_decoder_layers_vars[0],
        dim_feedforward_var=dim_feedforward_vars[0]
):
    nhead = nhead_var(emb_size)
    num_decoder_layers = num_decoder_layers_var(num_encoder_layers)
    dim_feedforward = dim_feedforward_var(emb_size)

    return dict(
        emb_size=emb_size,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        lr=lr
    )


# Argument parser for training
def parse_args(model_number):
    # Парсинг аргументов
    class CustomFormatter(argparse.HelpFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _get_help_string(self, action):
            help = action.help
            if '%(default)' in action.help or action.default is argparse.SUPPRESS:
                return help

            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                help += ' Значение по умолчанию: %(default)s.'
            return help

    formatter_class = lambda prog: CustomFormatter(prog, max_help_position=6, width=80)

    parser = argparse.ArgumentParser(
        description=f"Обучение и валидация модели №{model_number}.",
        formatter_class=formatter_class,
        add_help=False
    )

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Показать это сообщение с помощью и выйти.')

    parser.add_argument('-s', '--random-seed', type=int, default=179,
                        help='Зерно для генератора случайных чисел.')

    parser.add_argument('-f', '--frac', type=float, default=0.1,
                        help='Доля объектов, для которых считаются метрики на валидации.')

    parser.add_argument('-n', '--num_epochs', type=int, default=2,
                        help='Количество эпох обучения.')
    parser.add_argument('--num_epochs_lr_scheduler', type=int, default=2,
                        help='Полное количество эпох обучения (для сообщения lr scheduler`у).')

    parser.add_argument('-b', '--beam-width', type=int, default=1,
                        help='Ширина beam search для предсказывания ответа.')

    parser.add_argument('--train-utc', action='store_true',
                        help='Использование правильных классов при обучении (UTC — Use True Classes).')
    parser.add_argument('--val-utc', action='store_true',
                        help='Использование правильных классов при валидации (UTC — Use True Classes).')

    parser.add_argument('-v', '--val', action='store_true',
                        help='Валидировать модель, а не обучать.')

    args = parser.parse_args()

    return args
