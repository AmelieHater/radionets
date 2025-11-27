import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)
from lightning.pytorch.callbacks import Callback as LightningCallback
from lightning.pytorch.loggers import CometLogger
from matplotlib.colors import PowerNorm
from pydantic import BaseModel

from radionets.evaluation.utils import apply_symmetry, get_ifft
from radionets.plotting.utils import get_vmin_vmax, set_cbar


class Callbacks:
    @classmethod
    def get_callbacks(cls, train_config: BaseModel) -> list:
        default_callback = RichProgressBar()
        callbacks = [default_callback]

        if train_config.callbacks.model_checkpoint:
            model_checkpoint = ModelCheckpoint(
                **train_config.callbacks.model_checkpoint.model_dump()
            )
            callbacks.append(model_checkpoint)

        if train_config.callbacks.batch_size_finder:
            batch_size_finder = BatchSizeFinder(
                **train_config.callbacks.batch_size_finder.model_dump()
            )
            callbacks.append(batch_size_finder)

        if train_config.callbacks.early_stopping:
            early_stopping = EarlyStopping(
                **train_config.callbacks.early_stopping.model_dump()
            )
            callbacks.append(early_stopping)

        if train_config.callbacks.lr_monitor:
            lr_monitor = LearningRateMonitor(
                **train_config.callbacks.lr_monitor.model_dump()
            )
            callbacks.append(lr_monitor)

        if train_config.callbacks.device_stats_monitor:
            callbacks.append(DeviceStatsMonitor())

        if train_config.callbacks.timer:
            timer = Timer(**train_config.callbacks.timer.model_dump())
            callbacks.append(timer)

        if train_config.logging.comet_ml:
            callbacks.append(CometCallback(train_config))

        return callbacks


class CometCallback(LightningCallback):
    def __init__(self, train_config, *args, **kwargs):
        super().__init__()
        self.train_config = train_config
        self.amp_phase = train_config.general.amp_phase
        self.scale = train_config.logging.scale

        self.experiment = None
        self.cached_batch = None

        data_types = ["Amplitude", "Phase"] if self.amp_phase else ["Real", "Imaginary"]

        results = [" Prediction", " Ground Truth"]
        self.pred_plot_titles = [t + r for r in results for t in data_types]

    def plot_val_pred(self, predictions, targets, current_epoch: int):
        fig, axs = plt.subplots(
            2, 2, figsize=(12, 8.5), layout="constrained", sharex=True, sharey=True
        )
        axs = axs.flatten()

        limits_0 = get_vmin_vmax(targets[0, 0])  # Limits for amp/real
        limits_1 = get_vmin_vmax(targets[0, 1])  # Limits for phase/imaginary

        im0 = axs[0].imshow(
            predictions[0, 0],
            cmap="radionets.PuOr",
            vmin=-limits_0,
            vmax=limits_0,
            origin="lower",
        )
        im1 = axs[1].imshow(
            predictions[0, 1],
            cmap="radionets.PuOr",
            vmin=-limits_1,
            vmax=limits_1,
            origin="lower",
        )
        im2 = axs[2].imshow(
            targets[0, 0],
            cmap="radionets.PuOr",
            vmin=-limits_0,
            vmax=limits_0,
            origin="lower",
        )
        im3 = axs[3].imshow(
            targets[0, 1],
            cmap="radionets.PuOr",
            vmin=-limits_1,
            vmax=limits_1,
            origin="lower",
        )

        for ax, im, title in zip(
            axs,
            [im0, im1, im2, im3],
            self.pred_plot_titles,
        ):
            set_cbar(fig, ax, im, title=title, phase="Phase" in title)

        axs[0].set(ylabel="Frequels")
        axs[2].set(xlabel="Frequels", ylabel="Frequels")
        axs[3].set(xlabel="Frequels")

        self.experiment.log_figure(
            figure=fig, figure_name=f"fourier_pred_{current_epoch}"
        )

        plt.close(fig)

    def plot_val_fft(self, predictions, targets, current_epoch):
        ifft_pred = get_ifft(
            predictions,
            amp_phase=self.amp_phase,
            scale=self.scale,
        )
        ifft_truth = get_ifft(targets, amp_phase=self.amp_phase, scale=self.scale)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

        im0 = axs[0].imshow(
            ifft_pred,
            norm=PowerNorm(0.25, vmax=ifft_truth.max()),
            cmap="inferno",
            origin="lower",
        )
        im1 = axs[1].imshow(
            ifft_truth,
            norm=PowerNorm(0.25),
            cmap="inferno",
            origin="lower",
        )

        limits = get_vmin_vmax(ifft_pred - ifft_truth)
        im2 = axs[2].imshow(
            ifft_pred - ifft_truth,
            cmap="radionets.PuOr",
            vmin=-limits,
            vmax=limits,
            origin="lower",
        )

        for ax, im, title in zip(
            axs,
            [im0, im1, im2],
            ["Prediction", "Truth", "Difference"],
        ):
            set_cbar(fig, ax, im, title="FFT " + title)

        axs[0].set(
            ylabel="Pixels",
            xlabel="Pixels",
        )
        axs[1].set_xlabel("Pixels")
        axs[2].set_xlabel("Pixels")

        self.experiment.log_figure(figure=fig, figure_name=f"fft_pred_{current_epoch}")
        plt.close(fig)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        """Log predictions at validation epoch end."""

        if self.cached_batch is None:
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))

            # cache only one sample
            self.cached_batch = (
                batch[0][0][None, ...].cpu(),
                batch[1][0][None, ...].cpu(),
            )

        if self.experiment is None:
            try:
                self.experiment = next(
                    logger.experiment
                    for logger in trainer.loggers
                    if isinstance(logger, CometLogger)
                )
            except StopIteration as e:
                raise ValueError(
                    f"Could not find a CometLogger instance in {trainer.loggers}."
                ) from e

        if (trainer.current_epoch + 1) % self.train_config.logging.plot_n_epochs == 0:
            batch = (
                self.cached_batch[0].to(pl_module.device),
                self.cached_batch[1].to(pl_module.device),
            )

            predictions = pl_module.predict_step(batch, batch_idx=0).cpu()
            targets = batch[1].cpu()

            # check if images are half or full
            if predictions.shape[-2] != predictions.shape[-1]:
                predictions = apply_symmetry(
                    predictions,
                    overlap=self.train_config.dataloader.overlap,
                )
                targets = apply_symmetry(
                    targets,
                    overlap=self.train_config.dataloader.overlap,
                )

            self.plot_val_pred(
                predictions,
                targets,
                current_epoch=trainer.current_epoch,
            )

            self.plot_val_fft(
                predictions,
                targets,
                current_epoch=trainer.current_epoch,
            )
