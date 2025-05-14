import time
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Timer:
    _timings = defaultdict(list)

    def __init__(self, name: Optional[str] = None, accumulate: bool = False):
        self.name = name
        self.accumulate = accumulate
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        elapsed = time.perf_counter() - self.start_time
        if self.name:
            if self.accumulate:
                Timer._timings[self.name].append(elapsed)
            else:
                logger.info(
                    f"Timer '{self.name}' elapsed time: {elapsed:.4f} seconds")
        return elapsed

    @classmethod
    def summary(cls):
        for name, records in cls._timings.items():
            total_time = sum(records)
            count = len(records)
            avg = total_time / count if count > 0 else 0
            logger.info(
                f"Timer '{name}' - Total: {total_time:.4f} seconds, Count: {count}, Average: {avg:.4f} seconds")
