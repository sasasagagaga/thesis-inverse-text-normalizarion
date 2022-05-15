# Подбираем размер emb_size еще раз, так как оказалось, что лучше использовать dim_feedforward = emb_size


# Common libraries
import os
import sys
import argparse
from argparse import Namespace

# All local code
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from ITN.utils import seed_everything, get_device
from ITN.metrics import compute_asr_metrics
from ITN.models.trainer import Trainer
from ITN.models.seq2seq_transformer import (
    CLI_seq2seq_transformer,
    seq2seq_transformer_translate
)
from ITN.models.seq2seq_transformer_with_token_cls import (
    Seq2SeqTransformerWithEncoderForClasses,
)


tokenizer_name = "my_ru"
data_language = "ru"


# Configurable parameters
emb_size_vars = [128, 256, 512, 1024]

# Functionally configurable parameters
num_decoder_layers_vars = [
    lambda num_encoder_layers: num_encoder_layers,
]

# Chosen parameters
dim_feedforward_vars = [
    lambda emb_size: 1 * emb_size,  # NOTE: The best variant
]

nhead_vars = [lambda emb_size: emb_size // 64]  # NOTE: Вроде так всегда норм делать


# Configurator
def configure_model_params(
        *, emb_size=256, num_encoder_layers=3, lr=3e-4,
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


# Global test counter
test_num = 0


# Test runner
def run_test(
        *,
        emb_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        lr,
        num_epochs,
        num_epochs_lr_scheduler,
        validation
):
    global test_num

    model_version = (
        f"{34 + test_num}-ablation-"
        f"({emb_size}, {nhead}, {num_encoder_layers}, {num_decoder_layers}, {dim_feedforward})"
        f"-lr={lr}+linear_lr_scheduler"
    )

    print(f"\n\n***** {'Validate' if validation else 'Train'}: {model_version} *****\n")

    seed_everything(179)

    if validation:
        model_params = Namespace(
            model_class=Seq2SeqTransformerWithEncoderForClasses,
            model_version=model_version,
            epoch=num_epochs
        )
    else:
        model_params = Namespace(
            model_class=Seq2SeqTransformerWithEncoderForClasses,
            model_version=model_version,
            parameters=dict(
                transformer_encoder_params_for_load=dict(
                    model_version=3,
                    epoch=1,
                    language=data_language
                ),
                tokenizer_name=tokenizer_name,

                emb_size=emb_size,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
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

        optimizer_params=dict(lr=lr),
        lr_scheduler=False if validation else None,
        lr_scheduler_params=num_epochs_lr_scheduler
    )

    if validation:
        trainer.validate_epoch(fraction=0.1)
    else:
        trainer.train(
            num_epochs=num_epochs,
            write_every_iters=100,
            print_every_iters=200,
            compute_metrics_every_iters=200
        )

    test_num += 1


def main(validation=False):
    num_epochs = 2
    num_epochs_lr_scheduler = 2

    for emb_size in emb_size_vars:
        run_test(
            **configure_model_params(emb_size=emb_size),
            num_epochs=num_epochs,
            num_epochs_lr_scheduler=num_epochs_lr_scheduler,
            validation=validation
        )


if __name__ == "__main__":
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
        description='Ablation study для модели №1, 4 этап.',
        formatter_class=formatter_class,
        add_help=False
    )

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Показать это сообщение с помощью и выйти.')
    parser.add_argument('-v', '--val', action='store_true',
                        help='Валидировать модель, а не обучать.')

    args = parser.parse_args()

    main(validation=args.val)
