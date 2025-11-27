from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)
from pydantic import BaseModel


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

        return callbacks
