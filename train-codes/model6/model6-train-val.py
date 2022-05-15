# Обучаем модель №6 с подобранными ГП (по модели №1)


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

from ITN.models.seq2seq_transformer_with_token_cls import (
    Seq2SeqTransformerWithClassEmbeddingsGroupDecoderTrain,
    Seq2SeqTransformerWithClassEmbeddingsGroupDecoderEval,
    CLI_model6_train,
    model6_translate
)


tokenizer_name = "my_ru"
data_language = "ru"


# Global test counter
test_num = 5


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

    if not validation and train_utc or validation and val_utc:
        model_class = Seq2SeqTransformerWithClassEmbeddingsGroupDecoderTrain
        generate_batch_clb = lambda x, tokenizer: x
        calc_loss_iter_clb = CLI_model6_train
        translate_clb = model6_translate
    else:
        model_class = Seq2SeqTransformerWithClassEmbeddingsGroupDecoderEval
        generate_batch_clb = None
        calc_loss_iter_clb = None
        translate_clb = model6_translate

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

                ** model_params,
            ),
        )

    trainer = Trainer(
        model_params=model_params,
        data_target="seq2seq",
        calc_loss_iter_clb=calc_loss_iter_clb,
        compute_metrics_clb=compute_asr_metrics(skip_trans_examples=True),
        translate_clb=translate_clb(beam_width=args.beam_width),

        batch_size=16,
        device=get_device(),
        generate_batch_clb=generate_batch_clb,

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
        trainer.validate_epoch(fraction=args.frac)
    else:
        trainer.train(
            num_epochs=num_epochs,
            write_every_iters=100,
            print_every_iters=200,
            compute_metrics_every_iters=200
        )

    test_num += 1


def main():
    args = parse_args(model_number=6)
    print(args)

    if not args.train_utc:
        global test_num
        test_num += 1

    if not args.train_utc:
        raise ValueError("NOT SUPPORTED YET")

    run_test(
        model_params=configure_model_params(emb_size=512, num_encoder_layers=3, lr=1e-4),
        args=args
    )


if __name__ == "__main__":
    main()
