"""Blocking calls, run off the Qt thread and answered in it."""

from typing import Any, Callable, Optional

from koil import run_threaded
from koil.qt import async_to_qt
from qtpy import QtCore


class _Job(QtCore.QObject):
    """One call and who hears of its end."""

    finished = QtCore.Signal(object)

    def __init__(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        done: Optional[Callable[[Any], None]],
        failed: Optional[Callable[[BaseException], None]],
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self.done = done
        self.failed = failed

        async def call() -> Any:  # noqa: ANN401
            return await run_threaded(function, *args)

        self.task = async_to_qt(call, parent=self)
        self.task.returned.connect(self.on_returned)
        self.task.errored.connect(self.on_errored)
        self.task.cancelled.connect(self.on_errored)

    def on_returned(self, value: Any) -> None:  # noqa: ANN401
        self.finished.emit(self)
        if self.done is not None:
            self.done(value)

    def on_errored(self, error: BaseException) -> None:
        self.finished.emit(self)
        if self.failed is not None:
            self.failed(error)


class Jobs(QtCore.QObject):
    """Runs blocking functions in worker threads of the connected run.

    Each call is its own task, so two that overlap cannot answer each other's
    callers. The run has to be entered: the threads are its koil loop's.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._running: set[_Job] = set()

    def submit(
        self,
        function: Callable[..., Any],
        *args: Any,  # noqa: ANN401
        done: Optional[Callable[[Any], None]] = None,
        failed: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        """Call ``function(*args)`` off the Qt thread.

        Args:
            function: The blocking function.
            *args: Its arguments.
            done: Called in the Qt thread with what it returned.
            failed: Called in the Qt thread with what it raised.
        """
        job = _Job(function, args, done, failed, parent=self)
        job.finished.connect(self._forget)
        self._running.add(job)
        job.task.run()

    def _forget(self, job: _Job) -> None:
        self._running.discard(job)
        job.deleteLater()
