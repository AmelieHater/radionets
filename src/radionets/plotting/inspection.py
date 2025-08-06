from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from radionets.core.logging import setup_logger

LOGGER = setup_logger()


def plot_loss(learn, model_path: str | Path, output_format: str = "png") -> None:
    """
    Plot train and valid loss of model.

    Parameters
    ----------
    learn : learner-object
        learner containing data and model
    model_path : str
        path to trained model
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)

    save_path = model_path.with_suffix("")
    LOGGER.info(f"Plotting Loss for: {model_path.stem}")

    logscale = learn.avg_loss.plot_loss()
    title = str(model_path.stem).replace("_", " ")
    plt.title(rf"{title}")

    if logscale:
        plt.yscale("log")

    plt.savefig(
        f"{save_path}_loss.{output_format}", bbox_inches="tight", pad_inches=0.01
    )
    plt.clf()

    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr(learn, model_path: str | Path, output_format: str = "png") -> None:
    """
    Plot learning rate of model.

    Parameters
    ----------
    learn : learner-object
        learner containing data and model
    model_path : str or Path
        path to trained model
    output_format :
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)

    save_path = model_path.with_suffix("")
    LOGGER.info(f"Plotting Learning rate for: {model_path.stem}")

    learn.avg_loss.plot_lrs()

    plt.savefig(f"{save_path}_lr.{output_format}", bbox_inches="tight", pad_inches=0.01)
    plt.clf()

    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr_loss(
    learn, arch_name: str, out_path: str | Path, skip_last, output_format="png"
):
    """
    Plot loss of learning rate finder.

    Parameters
    ----------
    learn : learner-object
        learner containing data and model
    arch_path : str
        name of the architecture
    out_path : str
        path to save loss plot
    skip_last : int
        skip n last points
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)

    LOGGER.info(f"Plotting Lr vs Loss for architecture: {arch_name}")

    learn.recorder.plot_lr_find()
    out_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        out_path / f"lr_loss.{output_format}", bbox_inches="tight", pad_inches=0.01
    )

    mpl.rcParams.update(mpl.rcParamsDefault)
