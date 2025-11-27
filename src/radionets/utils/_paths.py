def _validate_pre_model_path(train_config):
    if not train_config.paths.pre_model:
        raise ValueError(
            f"'pre_model' path is {train_config.paths.pre_model} "
            "even though testing mode was started. Please make sure "
            "you provide a valid path to a model checkpoint file (.ckpt) "
            "in your configuration."
        )
    if not train_config.paths.pre_model.is_file():
        raise ValueError(
            f"'pre_model' path is {train_config.paths.pre_model}, "
            "but not a valid path to a model checkpoint file (.ckpt)."
        )
