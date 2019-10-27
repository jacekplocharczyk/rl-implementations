"""
Simple profiler for check how long function takes to finish.
WARNING: Shouldn't be used for nested functions (It changes performance dramaticaly).
"""

from collections import defaultdict
from functools import partial
import time
from typing import Any, Callable

import numpy as np


class Profiler:
    def __init__(self):
        self.checks = defaultdict(partial(np.ndarray, 0))

    def check(self, name: str, f: Callable) -> Callable:
        def profiled_func(*args, **kwargs) -> Any:
            start = time.perf_counter()
            out = f(*args, **kwargs)
            finish = time.perf_counter()
            t = finish - start
            self.checks[name] = np.append(self.checks[name], np.array([t]))
            return out

        return profiled_func

    def print(self):
        for k, v in self.checks.items():
            if len(v) > 0:
                print(
                    f"Name: {k:20}\tMean: {np.mean(v):>12.4}\tSTD: {np.std(v):>12.4}\t"
                    f"Calls: {len(v):>10}\tSummary Time: {len(v) * np.mean(v):>12.4}"
                )
