import os
from typing import Literal

from pydantic import BaseModel

__all__ = [
    "CSVLoggerConfig",
    "CometLoggerConfig",
    "MLFlowLoggerConfig",
]


class CSVLoggerConfig(BaseModel):
    """Lightning CSVLogger logging config"""

    name: str | None = "lightning_logs"
    version: int | str | None = None
    prefix: str = ""
    flush_logs_every_n_steps: int = 100


class CometLoggerConfig(BaseModel):
    """Lightning CometLogger logging config"""

    api_key: str | None = os.getenv("COMET_API_KEY")
    workspace: str | None = None
    experiment_key: str | None = None
    mode: Literal["get_or_create", "get", "create"] | None = None
    online: bool | None = None
    prefix: str | None = None


class MLFlowLoggerConfig(BaseModel):
    """Lightning MLFlowLogger logging config"""

    run_name: str | None = (None,)
    tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    tags: dict | None = None
    log_model: Literal[True, False, "all"] = False
    checkpoint_path_prefix: str = ""
    prefix: str = ""
    artifact_location: str | None = None
    run_id: str | None = None
    synchronous: bool | None = None
