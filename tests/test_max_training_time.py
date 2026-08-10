from training_code.callbacks import MaxTrainingTimeCallback


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTrainer:
    should_stop = False


def test_time_before_start_does_not_count() -> None:
    clock = FakeClock()
    callback = MaxTrainingTimeCallback(10.0, clock=clock)

    clock.now = 100.0
    callback.start()
    clock.now = 109.9

    assert not callback.has_expired()


def test_epoch_boundary_requests_stop_after_limit() -> None:
    clock = FakeClock()
    callback = MaxTrainingTimeCallback(10.0, clock=clock)
    trainer = FakeTrainer()

    callback.start()
    clock.now = 10.1
    callback.on_train_epoch_end(trainer, None)

    assert trainer.should_stop


def test_validation_boundary_requests_stop_after_limit() -> None:
    clock = FakeClock()
    callback = MaxTrainingTimeCallback(10.0, clock=clock)
    trainer = FakeTrainer()

    callback.start()
    clock.now = 10.0
    callback.on_validation_end(trainer, None)

    assert trainer.should_stop
