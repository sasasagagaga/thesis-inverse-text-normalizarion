
import os
import shutil
from argparse import Namespace
import logging
from typing import Union

import torch

from transformers import get_constant_schedule

from torch.utils.tensorboard import SummaryWriter

from ..tokenizer import decode
from ..data.data_loading import get_dataloaders, CLASS_PAD_IDX
from ..metrics import join_results
from ..core import language_for_human, run_logs_dir, tensorboard_logs_dir

from .core import construct_model_dir, load_model, train_and_validate_epoch, validate_epoch


class Trainer:
    LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def setup_logger(self):
        file_handler = logging.FileHandler(
            os.path.join(
                run_logs_dir,
                language_for_human[self.data_language],
                self.model_name_for_saving,
                f"{self.model_version}.log"
            ),
            "a"
        )
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        file_handler.setFormatter(formatter)

        log = logging.getLogger()  # root logger — Good to get it only once.
        for handler in log.handlers[:]:  # remove the existing file handlers
            if isinstance(handler, logging.FileHandler):
                log.removeHandler(handler)

        log.addHandler(file_handler)  # set the new handler
        log.setLevel(logging.INFO)  # set the log level to INFO, DEBUG as the default is ERROR

    def __init__(
            self,

            *,

            model_params: Union[dict, Namespace],
            data_target,

            calc_loss_iter_clb,
            compute_metrics_clb,
            translate_clb,

            batch_size,

            loss_fn=None,
            optimizer=None,
            optimizer_params: dict = None,
            lr_scheduler=None,
            lr_scheduler_params: dict = None,
            device=torch.device("cuda:0"),
            tokenizer_name="my_ru",
            data_language="ru",
            generate_batch_clb=None,  # equals to data_target if not provided
            join_results_clb=None,    # equals to code.metrics.join_results if not provided
            metrics_to_calc="all",
            write_every_iters=200,
            iters_for_val="uniform",
            print_every_iters=False,
            save_every_iters=1000,
            compute_metrics_every_iters=2000,
            val_dataloader_fraction: float = 1.0,
            disable_val_fraction_writer_info: bool = False
    ):
        if not isinstance(model_params, (dict, Namespace)):
            raise ValueError("model should be dict or Namespace")

        if isinstance(model_params, dict):
            model_params = Namespace(**model_params)

        if "model_version" not in model_params:
            raise ValueError("model_version is not specified")

        self.model_params = model_params

        self.data_target = data_target

        self.calc_loss_iter_clb = calc_loss_iter_clb
        self.compute_metrics_clb = compute_metrics_clb
        self.translate_clb = translate_clb

        self.batch_size = batch_size

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self.device = device

        self.tokenizer_name = tokenizer_name
        self.data_language = data_language

        self.generate_batch_clb = generate_batch_clb
        self.join_results_clb = join_results_clb
        self.metrics_to_calc = metrics_to_calc

        self.write_every_iters = write_every_iters
        self.iters_for_val = iters_for_val
        self.print_every_iters = print_every_iters
        self.save_every_iters = save_every_iters
        self.compute_metrics_every_iters = compute_metrics_every_iters

        self.val_dataloader_fraction = val_dataloader_fraction
        self.disable_val_fraction_writer_info = disable_val_fraction_writer_info

        self.epoch = None
        self.model_version = None
        self.writer = None
        self.dataloaders = None

        self.__real_init()

    def __real_init(self):
        # NOTE: Находится не в __init__, чтобы не ошибиться при обращении к переменным (self.var_name vs var_name)

        # For logging
        self.model_version = self.model_params.model_version
        attr_for_name = "model" if "model" in self.model_params else "model_class"
        self.model_name_for_saving = getattr(self.model_params, attr_for_name).name_for_saving

        self.setup_logger()

        self.epoch = self.model_params.epoch if "epoch" in self.model_params else 0

        if "model" in self.model_params:
            self.model = self.model_params.model
        elif "parameters" in self.model_params:
            self.model = self.model_params.model_class(**self.model_params.parameters)
            self.model.init_weights()
        else:
            # self.model.epoch — эпоха, на которой обучение еще не проводилось (нумерация с нуля).
            # То есть после нулевой (первой по счету) эпохи обучения model.epoch = 1.
            loaded_model = load_model(self.model_params.model_class, self.model_version, self.epoch,
                                      self.data_language, self.device, keep_optimizer=(self.optimizer is None))
            if self.optimizer is None:
                self.model, self.optimizer = loaded_model
            else:
                self.model = loaded_model

        self.model = self.model.to(self.device)

        # Optimizer setup
        if self.optimizer_params is None:
            self.optimizer_params = {}
        if self.optimizer is None:
            self.optimizer = self.model.configure_optimizer(**self.optimizer_params)

        if self.generate_batch_clb is None:
            self.generate_batch_clb = self.data_target

        if self.join_results_clb is None:
            self.join_results_clb = join_results

        self.tokenizer = self.model.tokenizer

        if self.loss_fn is None:
            if self.data_target == "seq2seq":
                self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            elif self.data_target == "classes":
                self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=CLASS_PAD_IDX)
            else:
                raise ValueError("loss_fn is not provided and it can be deduced from other arguments")

        # Loading data
        dataloaders = get_dataloaders(self.tokenizer_name, self.data_language, self.data_target,
                                      self.generate_batch_clb, self.batch_size,
                                      tokenizer=self.tokenizer)  # self.model.tokenizer
        self.dataloaders = {
            "train": dataloaders[0],
            "val": dataloaders[1]
        }

        # LR scheduler setup (находится тут, потому что нужно сначала загрузить данные)
        if self.lr_scheduler_params is None:
            self.lr_scheduler_params = {}
        elif isinstance(self.lr_scheduler_params, int):
            num_epochs = self.lr_scheduler_params
            num_steps_per_epoch = len(self.dataloaders["train"])
            num_steps = num_epochs * num_steps_per_epoch

            self.lr_scheduler_params = dict(
                num_warmup_steps=num_steps_per_epoch // 5,
                num_training_steps=num_steps,
            )
        if self.lr_scheduler is False:
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        elif self.lr_scheduler is None:
            self.lr_scheduler = self.model.configure_lr_scheduler(self.optimizer, **self.lr_scheduler_params)

        self.refresh_writer()

    @property
    def log_dir(self):
        return os.path.join(
            tensorboard_logs_dir,
            self.model.name_for_saving,
            str(self.model_version)
        )

    @property
    def model_dir(self):
        return construct_model_dir(self.data_language, self.model, self.model_version)

    def refresh_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def change_model_version(self, new_model_version: str):
        self.model_version = new_model_version
        self.setup_logger()
        self.refresh_writer()

    def refresh_state(self, new_model_version=None, remove_models_and_logs=False):
        if remove_models_and_logs:
            shutil.rmtree(self.log_dir, ignore_errors=True)
            shutil.rmtree(self.model_dir, ignore_errors=True)

        if new_model_version is not None:
            self.change_model_version(new_model_version)  # To update logger

        self.refresh_writer()

        self.model.init_weights()
        self.optimizer = self.model.configure_optimizer(**self.optimizer_params)
        self.lr_scheduler = self.model.configure_lr_scheduler(**self.lr_scheduler_params)
        self.epoch = 0

    def calc_loss(self, srcs, tgts):
        return self.calc_loss_iter_clb(srcs, tgts, self.model, self.tokenizer, self.device, self.loss_fn)

    @property
    def dataloader_train(self):
        return self.dataloaders["train"]

    @property
    def dataloader_val(self):
        return self.dataloaders["val"]

    def train_and_validate_epoch(self, write_every_iters=None, iters_for_val=None,
                                 print_every_iters=None, save_every_iters=None,
                                 compute_metrics_every_iters=None):
        if self.calc_loss_iter_clb is None:
            raise ValueError("Model can't be trained (no callback for calculating loss)")

        def new_val_if_not_none(val, new_val):
            return new_val if new_val is not None else val

        train_and_validate_epoch(
            self.model_version,
            self.epoch,
            self.model, self.calc_loss, self.compute_metrics_clb, self.translate_clb,
            self.join_results_clb, self.metrics_to_calc,
            self.data_language, self.tokenizer, self.device,
            self.dataloaders["train"], self.dataloaders["val"],
            self.optimizer,  # loss_fn,
            self.lr_scheduler,
            self.writer,
            new_val_if_not_none(self.write_every_iters, write_every_iters),
            new_val_if_not_none(self.iters_for_val, iters_for_val),
            new_val_if_not_none(self.print_every_iters, print_every_iters),  # =None
            new_val_if_not_none(self.save_every_iters, save_every_iters),
            new_val_if_not_none(self.compute_metrics_every_iters, compute_metrics_every_iters)
        )

        self.epoch += 1

    def train(self, num_epochs, **kwargs):
        logging.info(f"Train process started. Epoch of model was {self.epoch}.")
        try:
            for relative_epoch in range(num_epochs):
                self.train_and_validate_epoch(**kwargs)
        except KeyboardInterrupt:
            logging.info("Train process was interrupted by keyboard interrupt.")
        else:
            logging.info("Train process finished.")

    @property
    def last_train_epoch(self):
        # Номер последней эпохи, на которой обучение было закончено до конца
        return self.epoch - 1

    def validate_epoch(self, fraction=None, disable_fraction_writer_info=None):
        def new_val_if_not_none(val, new_val):
            return new_val if new_val is not None else val

        validate_epoch(
            self.last_train_epoch,
            self.model,
            None if self.calc_loss_iter_clb is None else self.calc_loss,
            self.compute_metrics_clb, self.translate_clb,
            self.join_results_clb, self.metrics_to_calc,
            self.tokenizer, self.device,
            self.dataloaders["val"],
            self.writer,
            new_val_if_not_none(self.val_dataloader_fraction, fraction),
            new_val_if_not_none(self.disable_val_fraction_writer_info, disable_fraction_writer_info)
        )

    def decode(self, sequences, batch_first=False, use_model_tokenizer=False):
        return decode(self.model.tokenizer if use_model_tokenizer else self.tokenizer,
                      sequences, batch_first=batch_first)

    def translate(self, srcs, **translate_kwargs):
        return self.translate_clb(self.model, self.tokenizer, self.device, srcs, **translate_kwargs)

    def translate_from_data(self, segment="train", return_sources=False, return_targets=True, **translate_kwargs):
        srcs, tgts = next(iter(self.dataloaders[segment]))

        result = []
        if return_sources:
            result.append(self.decode(srcs))

        result.append(self.translate(srcs, **translate_kwargs))

        if return_targets:
            result.append(self.decode(tgts))

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def calculate_metrics(self, srcs, tgts, **translate_kwargs):
        return self.compute_metrics_clb(self.model, self.tokenizer, self.device, self.translate_clb,
                                        srcs, tgts, self.metrics_to_calc, **translate_kwargs)
