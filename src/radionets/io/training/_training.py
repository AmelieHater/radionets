import inspect
from collections.abc import Callable
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radionets.architecture import loss

__all__ = ["LossConfig", "OptimizerConfig", "LRSchedulerConfig"]


class LossConfig(BaseModel):
    loss_func: str | Callable = torch.nn.MSELoss

    model_config = ConfigDict(extra="allow")

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
                f"Unknown optimizer: TrainConfig got {loss_func} but expected "
                f"one of {set(avail_loss_funcs)}!"
            ) from e

        return loss_func


class OptimizerConfig(BaseModel):
    optimizer: str | Callable = torch.optim.AdamW
    lr: float = Field(default=1e-3, gt=0.0)

    model_config = ConfigDict(extra="allow")

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
                f"Unknown optimizer: TrainConfig got {optimizer} but expected "
                f"one of {set(avail_optimizers)}!"
            ) from e

        return optimizer


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduling configuration."""

    lr_start: float = Field(default=1e-5, gt=0.0)
    lr_max: float = Field(default=1e-3, gt=0.0)
    lr_stop: float = Field(default=1e-7, gt=0.0)
    warmup_ratio: float = Field(default=0.25, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_lr_schedule(self) -> Self:
        """Validate learning rate schedule relationships."""
        if self.lr_max <= self.lr_start:
            raise ValueError(
                f"lr_max ({self.lr_max}) must be > lr_start ({self.lr_start})"
            )
        if self.lr_max <= self.lr_stop:
            raise ValueError(
                f"lr_max ({self.lr_max}) must be > lr_stop ({self.lr_stop})"
            )
        return self
