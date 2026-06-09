"""Minimal local fallback for environments without the external loguru package."""

import logging


class _LoggerShim:
    def __init__(self) -> None:
        self._logger = logging.getLogger("axis")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def add(self, *_args, **_kwargs):
        return None

    def remove(self, *_args, **_kwargs):
        for handler in list(self._logger.handlers)[1:]:
            self._logger.removeHandler(handler)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)


logger = _LoggerShim()
