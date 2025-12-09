from contextlib import contextmanager

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


__all__ = ["carbontracker"]


@contextmanager
def carbontracker(train_config):
    if _CODECARBON_AVAILABLE and train_config.logging.codecarbon:
        tracker = OfflineEmissionsTracker(
            **train_config.logging.codecarbon.model_dump()
        )
        try:
            yield tracker.start()
        finally:
            tracker.stop()
    else:
        yield None
