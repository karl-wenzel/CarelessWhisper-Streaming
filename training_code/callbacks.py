import time
from collections.abc import Callable

from pytorch_lightning.callbacks import Callback


class MaxTrainingTimeCallback(Callback):
    """Stop training between epochs once a wall-clock budget is exhausted."""

    def __init__(
        self,
        max_training_time: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.max_training_time = max_training_time
        self._clock = clock
        self._started_at: float | None = None
        self._stop_announced = False

    def start(self) -> None:
        self._started_at = self._clock()

    def has_expired(self) -> bool:
        return (
            self._started_at is not None
            and self._clock() - self._started_at >= self.max_training_time
        )

    def _request_stop_if_expired(self, trainer) -> None:
        if not self.has_expired():
            return

        trainer.should_stop = True
        if not self._stop_announced:
            elapsed = self._clock() - self._started_at
            print(
                "Maximum training time reached "
                f"({elapsed:.1f}s >= {self.max_training_time:.1f}s); "
                "stopping before the next training epoch."
            )
            self._stop_announced = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Requirement: an epoch already in progress may overrun the limit, but
        # another training epoch must never be started after it is detected.
        self._request_stop_if_expired(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:
        # Post-epoch validation counts toward the same budget and can itself
        # cross the threshold before the next training epoch begins.
        self._request_stop_if_expired(trainer)
