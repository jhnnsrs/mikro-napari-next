"""Blocking calls leave the Qt thread and are answered in it."""

import threading
import time

import pytest

pytest.importorskip("qtpy.QtWidgets")

from koil.qt import create_qt_koil  # noqa: E402
from qtpy import QtWidgets  # noqa: E402

from mikro_napari.widgets.jobs import Jobs  # noqa: E402


def wait_for(qapp, condition, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert condition(), "the job never answered"


@pytest.fixture
def jobs(qapp):
    """Jobs over a koil loop of their own: the loop a connected run would bring."""
    owner = QtWidgets.QWidget()
    koil = create_qt_koil(owner)
    yield Jobs(owner)
    koil.exit()


def test_a_job_runs_off_the_qt_thread_and_answers_in_it(qapp, jobs) -> None:
    ran_in: list[int] = []
    answered: list[tuple[int, int]] = []

    def work(a: int, b: int) -> int:
        ran_in.append(threading.get_ident())
        return a + b

    jobs.submit(work, 2, 3, done=lambda value: answered.append((value, threading.get_ident())))
    wait_for(qapp, lambda: answered)

    here = threading.get_ident()
    assert answered == [(5, here)] and ran_in != [here]
    qapp.processEvents()
    assert jobs._running == set()


def test_what_a_job_raises_reaches_its_caller(qapp, jobs) -> None:
    failed: list[BaseException] = []

    def work() -> None:
        raise ValueError("no")

    jobs.submit(work, failed=failed.append)
    wait_for(qapp, lambda: failed)

    assert [str(error) for error in failed] == ["no"]


def test_overlapping_jobs_answer_their_own_callers(qapp, jobs) -> None:
    answers: dict[str, str] = {}

    def work(name: str, seconds: float) -> str:
        time.sleep(seconds)
        return name

    jobs.submit(work, "slow", 0.2, done=lambda value: answers.setdefault("first", value))
    jobs.submit(work, "fast", 0.0, done=lambda value: answers.setdefault("second", value))
    wait_for(qapp, lambda: len(answers) == 2)

    assert answers == {"first": "slow", "second": "fast"}
