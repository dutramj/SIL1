# 3rd party libraries
import pandas as pd

# VAHSimulator library
from . import performance_decorator

class Recorder:
    def __init__(self):
        self._data = {}
        self._initialized = False

    @performance_decorator.time_execution_stats
    def append(self, row: dict):
        if not self._initialized:
            self._data = {key: [] for key in row}
            self._initialized = True

        for key, value in row.items():
            self._data[key].append(value)

    def to_df(self):
        if not self._data:
            raise ValueError("No data to convert.")

        return pd.DataFrame(self._data)

    def clear(self):
        self._data.clear()
        self._initialized = False

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))
