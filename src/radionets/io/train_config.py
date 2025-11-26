import inspect
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import torch.cuda
import torch.nn
import torch.optim
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radionets.architecture import archs, loss

from . import data
from ._callbacks import (
    BatchSizeFinderCallbackConfig,
    EarlyStoppingCallbackConfig,
    LearningRateMonitorCallbackConfig,
    ModelCheckpointCallbackConfig,
    TimerCallbackConfig,
)
from ._logging import (
    CometLoggerConfig,
    CSVLoggerConfig,
    MLFlowLoggerConfig,
)
from ._misc import (
    CodeCarbonEmissionTrackerConfig,
    DeepSpeedConfig,
    DeviceConfig,
)


class LoggingConfig(BaseModel):
    """Logging and experiment tracking configuration."""

    project_name: str = "Radionets"
    plot_n_epochs: int = Field(default=10, gt=0)
    scale: bool = True
    default_logger: CSVLoggerConfig = CSVLoggerConfig()
    comet_ml: bool | CometLoggerConfig = False
    mlflow: bool | MLFlowLoggerConfig = False

    @field_validator("default_logger", mode="after")
    @classmethod
    def validate_default_logger(cls, v):
        if isinstance(v, dict):
            return CSVLoggerConfig(**v)

        return v

    @field_validator("comet_ml", mode="after")
    @classmethod
    def validate_comet_ml(cls, v):
        if isinstance(v, dict):
            return CometLoggerConfig(**v)
        elif v is True:
            return CometLoggerConfig()  # Return defaults

        return v

    @field_validator("mlflow", mode="after")
    @classmethod
    def validate_mlflow(cls, v):
        if isinstance(v, dict):
            return MLFlowLoggerConfig(**v)
        elif v is True:
            return MLFlowLoggerConfig()  # Return defaults

        return v


class CallbacksConfig(BaseModel):
    "Callbacks configuration."

    model_checkpoint: bool | ModelCheckpointCallbackConfig = False
    batch_size_finder: bool | BatchSizeFinderCallbackConfig = False
    early_stopping: bool | EarlyStoppingCallbackConfig = False
    lr_monitor: bool | LearningRateMonitorCallbackConfig = False
    timer: bool | TimerCallbackConfig = False
    device_stats_monitor: bool = False

    @field_validator("model_checkpoint", mode="after")
    @classmethod
    def validate_model_checkpoint(cls, v):
        if isinstance(v, dict):
            return ModelCheckpointCallbackConfig(**v)
        elif v is True:
            return ModelCheckpointCallbackConfig()  # Return defaults

        return v

    @field_validator("batch_size_finder", mode="after")
    @classmethod
    def validate_batch_size_finder(cls, v):
        if isinstance(v, dict):
            return BatchSizeFinderCallbackConfig(**v)
        elif v is True:
            return BatchSizeFinderCallbackConfig()  # Return defaults

        return v

    @field_validator("early_stopping", mode="after")
    @classmethod
    def validate_early_stopping(cls, v):
        if isinstance(v, dict):
            return EarlyStoppingCallbackConfig(**v)
        elif v is True:
            return EarlyStoppingCallbackConfig()  # Return defaults

        return v

    @field_validator("lr_monitor", mode="after")
    @classmethod
    def validate_lr_monitor(cls, v):
        if isinstance(v, dict):
            return LearningRateMonitorCallbackConfig(**v)
        elif v is True:
            return LearningRateMonitorCallbackConfig()  # Return defaults

        return v

    @field_validator("timer", mode="after")
    @classmethod
    def validate_timer(cls, v):
        if isinstance(v, dict):
            return TimerCallbackConfig(**v)
        elif v is True:
            return TimerCallbackConfig()  # Return defaults

        return v


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_path: Path = Path("./example_data/")
    model_path: Path = Path("./build/example_model/example.model")
    pre_model: Path | None | Literal[False] = None

    @field_validator("data_path", "model_path", "pre_model")
    @classmethod
    def expand_path(cls, v: Path) -> Path:
        """Expand and resolve paths."""

        if v in {None, False}:
            v = None
        else:
            v.expanduser().resolve()

        return v


class GeneralConfig(BaseModel):
    """General training configuration."""

    arch_name: str | Callable = archs.SRResNet18
    loss_func: str | Callable = torch.nn.MSELoss
    optimizer: str | Callable = torch.optim.AdamW
    num_epochs: int = Field(default=5, gt=0)
    output_format: Literal["png", "jpg", "pdf", "svg"] = "png"
    amp_phase: bool = True
    normalize: bool = False
    source_list: bool = False
    inspection: bool = True
    switch_loss: bool = False
    when_switch: int = Field(default=25, ge=0)

    @field_validator("arch_name")
    @classmethod
    def load_arch_instance(cls, arch: str):
        avail_archs = {}

        for member in inspect.getmembers(archs):
            if inspect.isclass(member[1]):
                avail_archs[member[0]] = member[1]

        try:
            arch = avail_archs[arch]
        except KeyError as e:
            raise ValueError(
                f"unkown architecture: TrainConfig got {arch} but expected "
                f"one of {avail_archs.keys()}!"
            ) from e

        return arch

    @field_validator("optimizer")
    @classmethod
    def load_optimizer_instance(cls, optimizer: str):
        avail_optimizers = {}

        for member in inspect.getmembers(torch.optim):
            if inspect.isclass(member[1]):
                avail_optimizers[member[0]] = member[1]

        try:
            optimizer = avail_optimizers[optimizer]
        except KeyError as e:
            raise ValueError(
                f"unkown optimizer: TrainConfig got {optimizer} but expected "
                f"one of {set(avail_optimizers)}!"
            ) from e

        return optimizer

    @field_validator("loss_func")
    @classmethod
    def load_loss_func_instance(cls, loss_func: str):
        if isinstance(loss_func, type):
            return loss_func

        avail_loss_funcs = {}

        for member in inspect.getmembers(torch.nn):
            if inspect.isclass(member[1]):
                avail_loss_funcs[member[0]] = member[1]

        for member in inspect.getmembers(loss):
            if inspect.isclass(member[1]):
                avail_loss_funcs[member[0]] = member[1]

        try:
            loss_func = avail_loss_funcs[loss_func]
        except KeyError as e:
            raise ValueError(
                f"unkown optimizer: TrainConfig got {loss_func} but expected "
                f"one of {set(avail_loss_funcs)}!"
            ) from e

        return loss_func


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""

    datamodule: str | Callable = data.H5DataModule
    fourier: bool = True
    num_workers: int = Field(default=10, gt=0)

    model_config = ConfigDict(extra="allow")

    @field_validator("datamodule")
    @classmethod
    def load_data_module_instance(cls, name: str):
        if isinstance(name, type):
            return name

        avail_data_modules = {}

        for member in inspect.getmembers(data):
            if inspect.isclass(member[1]):
                avail_data_modules[member[0]] = member[1]

        try:
            data_module = avail_data_modules[name]
        except KeyError as e:
            raise ValueError(
                f"unkown optimizer: TrainConfig got {name} but expected "
                f"one of {set(avail_data_modules)}!"
            ) from e

        return data_module


class HypersConfig(BaseModel):
    """Hyperparameters configuration."""

    batch_size: int = Field(default=100, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)


class ParamSchedulingConfig(BaseModel):
    """Learning rate scheduling configuration."""

    use: bool = True
    lr_start: float = Field(default=7e-2, gt=0.0)
    lr_max: float = Field(default=3e-1, gt=0.0)
    lr_stop: float = Field(default=5e-2, gt=0.0)
    lr_ratio: float = Field(default=0.25, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_lr_schedule(self) -> Self:
        """Validate learning rate schedule relationships."""
        if self.use:
            if self.lr_max <= self.lr_start:
                raise ValueError(
                    f"lr_max ({self.lr_max}) must be > lr_start ({self.lr_start})"
                )
            if self.lr_max <= self.lr_stop:
                raise ValueError(
                    f"lr_max ({self.lr_max}) must be > lr_stop ({self.lr_stop})"
                )
        return self


class TrainConfig(BaseModel):
    """Main training configuration."""

    title: str = "Train configuration"
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    hypers: HypersConfig = Field(default_factory=HypersConfig)
    param_scheduling: ParamSchedulingConfig = Field(
        default_factory=ParamSchedulingConfig
    )
    deepspeed: bool | str | DeepSpeedConfig = False
    codecarbon: bool | CodeCarbonEmissionTrackerConfig = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()

    @field_validator("deepspeed", mode="after")
    @classmethod
    def validate_deepspeed(cls, v: bool | str | DeepSpeedConfig):
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return DeepSpeedConfig(**v)
        elif v is True:
            return DeepSpeedConfig()

        return v

    @field_validator("codecarbon", mode="after")
    @classmethod
    def validate_codecarbon(cls, v: bool | CodeCarbonEmissionTrackerConfig):
        if isinstance(v, dict):
            return CodeCarbonEmissionTrackerConfig(
                **v, project_name=cls.logging.project_name
            )
        elif v is True:
            return CodeCarbonEmissionTrackerConfig(
                project_name=cls.logging.project_name
            )

        # CometML automatically logs with codecarbon
        # if codecarbon is installed. This should ensure
        # that codecarbon is only used when set to in the config
        os.environ["COMET_AUTO_LOG_CO2"] = "false"

        return v
