
# *** Метрики *** #

# Common libraries
from deprecated import deprecated

import numpy as np

# Metrics
from jiwer import compute_measures
from ITN.data.data_loading import CLASS_PAD_IDX
from .tokenizer import decode
from ITN.utils import make_fp_function


ALL_ASR_METRICS = ("wer", "mer", "wil", "wip")
ALL_CLS_METRICS = ("accuracy",)


def join_results(results, metrics="all"):
    if isinstance(metrics, str):
        if metrics == "all":
            metrics = next(iter(results)).keys()
        elif metrics == "asr":
            metrics = ALL_ASR_METRICS
        elif metrics == "cls":
            metrics = ALL_CLS_METRICS

    return {metric: np.mean([result[metric] for result in results]) for metric in metrics}


def batched_asr_metrics(hypos, truths, metrics, skip_trans_examples=False):
    results = [
        compute_measures(truth, hypothesis)
        for truth, hypothesis in zip(truths, hypos)
        if truth.replace(' ', '') != "" and not (skip_trans_examples and "_trans" in hypothesis)
    ]
    return join_results(results, metrics)


def batched_wer(hypos, truths):
    return batched_asr_metrics(hypos, truths, metrics=("wer",))["wer"]


@make_fp_function
def compute_asr_metrics(model, tokenizer, device, translate_clb, srcs, tgts,
                        metrics="all", skip_trans_examples=False, **translate_kwargs):
    """In `translate_kwargs` provide additional arguments for `translate_clb`"""
    if isinstance(metrics, str) and metrics == "all":
        metrics = ALL_ASR_METRICS
    hypos = translate_clb(model, tokenizer, device, srcs, **translate_kwargs)
    truths = decode(tokenizer, tgts)
    return batched_asr_metrics(hypos, truths, metrics, skip_trans_examples)


@deprecated("This function is rotten, use `join_results` instead")
def join_results_for_classes(results, metrics="cls"):
    return join_results(results, metrics)


def batched_cls_metrics(hypos, truths, metrics="all"):
    if isinstance(metrics, str):
        if metrics != "all":
            raise ValueError("If 'metrics' is str, then it should be equal to 'all'")
        metrics = ALL_CLS_METRICS

    hypos = hypos.argmax(dim=-1)
    mask = truths != CLASS_PAD_IDX

    results = {
        "accuracy": (truths[mask] == hypos[mask]).float().mean().item()
    }

    return {metric: results[metric] for metric in metrics}


def compute_cls_metrics(model, tokenizer, device, translate_function, srcs, tgts,
                        metrics="cls", **translate_kwargs):
    hypos = translate_function(model, tokenizer, device, srcs, **translate_kwargs)
    return batched_cls_metrics(hypos, tgts, metrics)
