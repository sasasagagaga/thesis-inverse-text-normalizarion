# Обучаем модель №1 с подобранными ГП


# Common libraries
import os
import sys
from argparse import Namespace

# All local code
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from ITN.utils import seed_everything, get_device
from ITN.metrics import compute_asr_metrics
from ITN.models.trainer import Trainer
from ITN.models.train_utils import configure_model_params, parse_args

from ITN.models.seq2seq_transformer import (
    CLI_seq2seq_transformer,
    seq2seq_transformer_translate
)
from ITN.models.seq2seq_transformer_with_token_cls import (
    Seq2SeqTransformerWithEncoderForClasses
)


tokenizer_name = "my_ru"
data_language = "ru"

# UTC — Use True Classes, UPC — use predicted classes


# Global test counter
test_num = 36


def run_test(*, model_params, args):
    global test_num

    lr = model_params["lr"]
    model_params.pop("lr")  # To pass model_params to model class

    num_epochs = args.num_epochs
    num_epochs_lr_scheduler = args.num_epochs_lr_scheduler
    validation = args.val
    train_utc = args.train_utc
    val_utc = args.val_utc

    model_version = (
        f"{test_num}"
        f"-standard"
        f"-seed={args.random_seed}"
        f"-train={'UTC' if train_utc else 'UPC'}"
    )
    # UTC — Use True Classes, UPC — use predicted classes

    print(f"\n\n***** {'Validate' if validation else 'Train'}: {model_version} *****\n")

    seed_everything(args.random_seed)

    model_class = Seq2SeqTransformerWithEncoderForClasses

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
                transformer_encoder_params_for_load=dict(
                    model_version=3,
                    epoch=1,
                    language=data_language
                ),
                tokenizer_name=tokenizer_name,

                **model_params,
            ),
        )

    trainer = Trainer(
        model_params=model_params,
        data_target="seq2seq",
        calc_loss_iter_clb=CLI_seq2seq_transformer,
        compute_metrics_clb=compute_asr_metrics(skip_trans_examples=True),
        translate_clb=seq2seq_transformer_translate(beam_width=args.beam_width),

        batch_size=16,
        device=get_device(),

        optimizer_params=dict(lr=lr),
        lr_scheduler=False if validation else None,
        lr_scheduler_params=num_epochs_lr_scheduler
    )

    if validation:
        model_version += f"-val={'UTC' if val_utc else 'UPC'}"

    if args.beam_width is not None:
        model_version += f"-bw={args.beam_width}"

    print(model_class, model_version)

    if validation:
        trainer.change_model_version(model_version)
        trainer.validate_epoch(fraction=0.1)
    else:
        trainer.train(
            num_epochs=num_epochs,
            write_every_iters=100,
            print_every_iters=200,
            compute_metrics_every_iters=200
        )

    test_num += 1


def main():
    args = parse_args(model_number=1)
    print(args)

    if args.train_utc:
        raise ValueError("Not supported yet...")

    if args.train_utc:
        global test_num
        test_num += 1

    run_test(
        model_params=configure_model_params(emb_size=512, num_encoder_layers=3, lr=3e-4),
        args=args
    )


if __name__ == "__main__":
    main()
