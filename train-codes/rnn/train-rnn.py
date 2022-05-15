# Common libraries
from argparse import Namespace

# All local code
import ITN
from ITN.models.trainer import Trainer
from ITN.models.seq2seq_rnn import (
    Seq2SeqRNN, seq2seq_rnn_translate, CLI_seq2seq_rnn
)


tokenizer_name = "my_ru"
data_language = "ru"

model_params = Namespace(
    model_class=Seq2SeqRNN,
    model_version="17-softmax-removed-test-2",
    epoch=9
)

# OR

# model_params = Namespace(
#     model_class=Seq2SeqRNN,
#     model_version="17-softmax-removed-test-2",
#     parameters=dict(
#         tokenizer_name=tokenizer_name,
#         hidden_size=512,
#         num_layers=2,
#         dropout_p=0.1,
#         use_attn=True,
#     ),
# )

trainer = Trainer(
    model_params,
    "seq2seq",
    CLI_seq2seq_rnn,
    ITN.metrics.compute_asr_metrics,
    seq2seq_rnn_translate,
    batch_size=64
)

for _ in range(1):
    trainer.train_and_validate_epoch(
        write_every_iters=100,
        print_every_iters=200,
        compute_metrics_every_iters=200
    )

# trainer.train_and_validate_epoch()
