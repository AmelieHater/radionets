from pathlib import Path

import torch
from torch import nn

from radionets.core.logging import setup_logger

__all__ = [
    "init_cnn",
    "load_pre_model",
    "save_model",
    "symmetry",
]

LOGGER = setup_logger()


def _init_cnn(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, "bias", None) is not None:
            m.bias.data.zero_()
    for c in m.children():
        _init_cnn(c, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    _init_cnn(m, f)


def load_pre_model(learn, pre_path, visualize=False, plot_loss=False):
    """Loads a previously saved model as pre-model.

    Parameters
    ----------
    learn : learner
        Object of type learner.
    pre_path : str
        Path to the pre-model.
    visualize : bool
        Default: False
    plot_loss : bool
        Default: False
    """
    name_pretrained = Path(pre_path).stem
    LOGGER.info(f"Load pretrained model: {name_pretrained}")

    if torch.cuda.is_available() and not plot_loss:
        checkpoint = torch.load(pre_path)
    else:
        checkpoint = torch.load(pre_path, map_location=torch.device("cpu"))

    if visualize:
        learn.load_state_dict(checkpoint["model"])
        return checkpoint["norm_dict"]
    elif plot_loss:
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
    else:
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]


def save_model(learn, model_path):
    if hasattr(learn, "normalize"):
        if learn.normalize.mode == "mean":
            norm_dict = {
                "mean_real": learn.normalize.mean_real,
                "mean_imag": learn.normalize.mean_imag,
                "std_real": learn.normalize.std_real,
                "std_imag": learn.normalize.std_imag,
            }
        elif learn.normalize.mode == "max":
            norm_dict = {"max_scaling": 0}
        elif learn.normalize.mode == "all":
            norm_dict = {"all": 0}
        else:
            raise ValueError(f"Undefined mode {learn.normalize.mode}, check for typos")
    else:
        norm_dict = {}

    torch.save(
        {
            "model": learn.model.state_dict(),
            "opt": learn.opt.state_dict(),
            "epoch": learn.epoch,
            "iters": learn.recorder.iters,
            "vals": learn.recorder.values,
            "train_loss": learn.avg_loss.loss_train,
            "valid_loss": learn.avg_loss.loss_valid,
            "lrs": learn.avg_loss.lrs,
            "norm_dict": norm_dict,
        },
        model_path,
    )


def symmetry(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError("The symmetry function only works for even image sizes.")
    upper_half = x[:, :, 0 : x.shape[2] // 2, :].clone()
    upper_left = upper_half[:, :, :, 0 : upper_half.shape[3] // 2].clone()
    upper_right = upper_half[:, :, :, upper_half.shape[3] // 2 :].clone()
    a = torch.flip(upper_left, dims=[-2, -1])
    b = torch.flip(upper_right, dims=[-2, -1])

    upper_half[:, :, :, 0 : upper_half.shape[3] // 2] = b
    upper_half[:, :, :, upper_half.shape[3] // 2 :] = a

    x[:, 0, x.shape[2] // 2 :, :] = upper_half[:, 0]
    x[:, 1, x.shape[2] // 2 :, :] = -upper_half[:, 1]
    return x
