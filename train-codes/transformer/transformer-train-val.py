# Обучаем базовый трансформер с подобранными ГП (по модели №1)


# Common libraries
import os
import sys
import argparse
from argparse import Namespace

# All local code
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import configure_model_params

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from ITN.utils import seed_everything, get_device
from ITN.metrics import compute_asr_metrics
from ITN.models.trainer import Trainer

from ITN.models.seq2seq_transformer import (
    Seq2SeqTransformer,
    CLI_seq2seq_transformer,
    seq2seq_transformer_translate
)


tokenizer_name = "my_ru"
data_language = "ru"


# Global test counter
test_num = 6


def run_test(
        *,
        emb_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        lr,
        args
):
    global test_num

    model_class = Seq2SeqTransformer

    num_epochs = args.num_epochs
    num_epochs_lr_scheduler = args.num_epochs_lr_scheduler
    validation = args.val

    model_version = (
        f"{test_num}-"
        f"({emb_size}, {nhead}, {num_encoder_layers}, {num_decoder_layers}, {dim_feedforward})"
        f"-lr={lr}+linear_lr_scheduler"
    )
    print(f"\n\n***** {'Validate' if validation else 'Train'}: {model_version} *****\n")

    seed_everything(179)

    if validation:
        model_params = Namespace(
            model_class=model_class,
            model_version=model_version,
            epoch=num_epochs
        )
    else:
        model_params = Namespace(
            model_class=model_class,
            model_version=model_version,
            parameters=dict(
                tokenizer_name=tokenizer_name,

                emb_size=emb_size,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward
            ),
        )

    trainer = Trainer(
        model_params=model_params,
        data_target="seq2seq",
        calc_loss_iter_clb=CLI_seq2seq_transformer,
        compute_metrics_clb=compute_asr_metrics(skip_trans_examples=True),
        translate_clb=seq2seq_transformer_translate,

        batch_size=16,
        device=get_device(),
        generate_batch_clb=None,

        optimizer_params=dict(lr=lr),
        lr_scheduler=False if validation else None,
        lr_scheduler_params=num_epochs_lr_scheduler
    )

    if validation:
        trainer.validate_epoch(fraction=args.frac)
    else:
        trainer.train(
            num_epochs=num_epochs,
            write_every_iters=100,
            print_every_iters=200,
            compute_metrics_every_iters=200
        )

    test_num += 1


def main(args):
    run_test(
        **configure_model_params(),
        args=args
    )


def parse_args_and_run():
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
        description='Обучение и валидация модели №5.',
        formatter_class=formatter_class,
        add_help=False
    )

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Показать это сообщение с помощью и выйти.')

    parser.add_argument('-f', '--frac', type=float, default=0.1,
                        help='Доля объектов, для которых считаются метрики на валидации.')

    parser.add_argument('-n', '--num_epochs', type=int, default=2,
                        help='Количество эпох обучения.')
    parser.add_argument('--num_epochs_lr_scheduler', type=int, default=2,
                        help='Полное количество эпох обучения (для сообщения lr scheduler`у).')

    parser.add_argument('-v', '--val', action='store_true',
                        help='Валидировать модель, а не обучать.')

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    parse_args_and_run()
