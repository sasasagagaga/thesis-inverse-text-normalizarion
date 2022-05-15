# Подбираем размеры emb_size, количество слоев в трансформере и lr.
# Обучение моделей.


# Common libraries
import os
import sys
from argparse import Namespace

# All local code
sys.path.insert(1, os.path.join(sys.path[0], '..'))

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
emb_size_vars = [128, 256, 512]
num_encoder_layers_vars = [2, 4, 6]
lr_vars = [3e-4, 1e-3]

# Functionally configurable parameters
nhead_vars = [lambda emb_size: emb_size // 64]
num_decoder_layers_vars = [
    lambda num_encoder_layers: num_encoder_layers,
    # lambda num_encoder_layers: num_encoder_layers + 1
]
dim_feedforward_vars = [lambda emb_size: 4 * emb_size]
# TODO: мб все-таки лучше [lambda emb_size: emb_size]?


# Configurator
def configure_model_params(*, emb_size=256, num_encoder_layers=3, lr=1e-4):
    nhead = nhead_vars[0](emb_size)
    num_decoder_layers = num_decoder_layers_vars[0](num_encoder_layers)
    dim_feedforward = dim_feedforward_vars[0](emb_size)

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
        num_epochs
):
    global test_num

    model_version = (
        f"{16 + test_num}-ablation-"
        f"({emb_size}, {nhead}, {num_encoder_layers}, {num_decoder_layers}, {dim_feedforward})"
        f"-lr={lr}"
    )

    print(f"\n\n***** {model_version} *****\n")

    seed_everything(179)

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
        lr_scheduler=False
    )

    trainer.train(
        num_epochs=num_epochs,
        write_every_iters=100,
        print_every_iters=200,
        compute_metrics_every_iters=200
    )

    test_num += 1


def main():
    num_epochs = 1

    for num_encoder_layers in num_encoder_layers_vars:
        run_test(
            **configure_model_params(num_encoder_layers=num_encoder_layers),
            num_epochs=num_epochs
        )

    for lr in lr_vars:
        run_test(
            **configure_model_params(lr=lr),
            num_epochs=num_epochs
        )

    for emb_size in emb_size_vars:
        run_test(
            **configure_model_params(emb_size=emb_size),
            num_epochs=num_epochs
        )


if __name__ == "__main__":
    main()
