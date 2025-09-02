# creates chronological folds with a train block 
# -> gap (purge) -> validation block pattern. 
# No shuffling, no leakage.
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_range: Tuple[int, int]
    gap_range: Tuple[int, int]
    val_range: Tuple[int, int]

class PurgedTimeSeriesSplit:
    """
    Chronological TSCV with an explicit 'purge' gap between train and validation.
    Each split yields (train_idx, val_idx), where indices refer to the rows of the
    input time-ordered array/dataframe.

    Example geometry (indices increase with time):

        [------- TRAIN -------][--- GAP ---][--- VALIDATION ---]

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    val_size : int
        Number of samples (days) in each validation block.
    gap_size : int
        Purge length between the end of train and the start of validation.
    min_train_size : int
        Minimum number of samples required in each training block.
    step : int, optional
        How far to advance the validation window between folds. Defaults to val_size
        (non-overlapping validation). Use a smaller step for overlapping val windows.
    """

    def __init__(
        self,
        n_splits: int,
        val_size: int,
        gap_size: int,
        min_train_size: int,
        step: int | None = None,
    ) -> None:
        assert n_splits >= 1
        assert val_size >= 1
        assert gap_size >= 0
        assert min_train_size >= 1
        self.n_splits = n_splits
        self.val_size = val_size
        self.gap_size = gap_size
        self.min_train_size = min_train_size
        self.step = step or val_size

    def split(self, X: pd.DataFrame | np.ndarray) -> Iterator[Fold]:
        n = len(X)
        # Determine the last possible validation window start
        total_val_span = self.val_size + 0  # explicit
        # we will slide validation window from left to right
        starts: List[int] = []
        # start candidate after min_train_size + gap
        start_min = self.min_train_size + self.gap_size
        # last valid start so that start+val_size-1 <= n-1
        start_max = n - self.val_size
        if start_min > start_max:
            raise ValueError("Not enough data to create at least one fold with given sizes.")
        s = start_min
        while s <= start_max and len(starts) < self.n_splits:
            starts.append(s)
            s += self.step

        if len(starts) < self.n_splits:
            # fall back: take the last n_splits windows ending at the end
            starts = list(range(max(start_min, n - self.val_size - self.step*(self.n_splits-1)),
                                n - self.val_size + 1, self.step))[:self.n_splits]

        for v_start in starts:
            v_end = v_start + self.val_size - 1
            gap_end = v_start - 1
            gap_start = max(self.min_train_size, v_start - self.gap_size)
            # Train is everything strictly before gap_start
            t_end = gap_start - 1
            t_start = 0
            if t_end - t_start + 1 < self.min_train_size:
                # not enough train data, skip
                continue

            train_idx = np.arange(t_start, t_end + 1, dtype=int)
            gap_idx = np.arange(gap_start, gap_end + 1, dtype=int) if gap_end >= gap_start else np.array([], dtype=int)
            val_idx = np.arange(v_start, v_end + 1, dtype=int)

            yield Fold(
                train_idx=train_idx,
                val_idx=val_idx,
                train_range=(t_start, t_end),
                gap_range=(gap_start, gap_end if len(gap_idx) else gap_start - 1),
                val_range=(v_start, v_end),
            )

def attach_order(df: pd.DataFrame, date_col: str = "date_id") -> pd.DataFrame:
    """
    Ensure chronological order by date_col ascending, and reset index to 0..n-1.
    Returns a copy with 'row_id' column equal to the new position.
    """
    df = df.sort_values(date_col).reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df), dtype=int)
    return df
