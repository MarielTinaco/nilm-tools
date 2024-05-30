
import numpy as np
import pytest

from adinilm.data_processing.sequence_scanning import *


def test_window_Sequence_scan_init():
        scanner = SequenceScannerContext()

        assert isinstance(scanner.strategy, WindowSequenceScanner)


@pytest.mark.parametrize(
        "dummy_data, seq_len, n_windows",
        [(np.zeros(shape=(1000,), dtype=float), 100, 10),
         (np.ones(shape=(1000,), dtype=float), 100, 10),
         (np.zeros(shape=(100,), dtype=float), 100, 1),
         (np.ones(shape=(100,), dtype=float), 100, 1),
         (np.zeros(shape=(10,), dtype=float), 100, 1),
         (np.ones(shape=(10,), dtype=float), 100, 1),]
)
def test_window_sequence_scan_shape(dummy_data, seq_len, n_windows):
        strategy = WindowSequenceScanner(seq_len=seq_len, n_windows=n_windows)
        scanner = SequenceScannerContext(strategy=strategy)
        output_data = scanner(dummy_data)

        assert len(output_data) == n_windows

        for window in output_data:
                assert window.shape == (seq_len,)

def test_odd_window_sequence_scan_init():
        scanner = SequenceScannerContext(OddWindowSequenceScanner(100))

        assert isinstance(scanner.strategy, OddWindowSequenceScanner)

        pop = scanner(np.zeros(shape=(1000,), dtype=float))

        assert pop[0].shape == (99,)

@pytest.mark.parametrize(
        "dummy_data, seq_len, n_windows",
        [(np.zeros(shape=(100,), dtype=float), 100, 10),
         (np.ones(shape=(100,), dtype=float), 100, 10),
         (np.zeros(shape=(10,), dtype=float), 100, 10),
         (np.ones(shape=(10,), dtype=float), 100, 10),]
)
def test_sliding_short_window_sequence_scan_shape(dummy_data, seq_len, n_windows):
        strategy = SlidingShortWindowSequenceScanner(seq_len=seq_len, n_windows=n_windows)
        scanner = SequenceScannerContext(strategy=strategy)
        output_data = scanner(dummy_data)

        assert len(output_data) == n_windows

        for window in output_data:
                assert window.shape == (seq_len,)

@pytest.mark.parametrize(
        "dummy_data, seq_len, n_windows",
        [(np.zeros(shape=(100,), dtype=float), 100, 10),
         (np.ones(shape=(100,), dtype=float), 100, 10),
         (np.zeros(shape=(10,), dtype=float), 100, 10),
         (np.ones(shape=(10,), dtype=float), 100, 10),
         (np.zeros(shape=(1000,), dtype=float), 100, 10),
         (np.ones(shape=(1000,), dtype=float), 100, 10),
         (np.zeros(shape=(100,), dtype=float), 100, 1),
         (np.ones(shape=(100,), dtype=float), 100, 1)]
)
def test_adaptive_window_sequence_scan_shape(dummy_data, seq_len, n_windows):
        strategy = LengthAdaptiveSequenceScanner(seq_len=seq_len, n_windows=n_windows)
        scanner = SequenceScannerContext(strategy=strategy)
        output_data = scanner(dummy_data)

        assert len(output_data) == n_windows

        for window in output_data:
                assert window.shape == (seq_len,)

@pytest.mark.parametrize(
        "dummy_data, seq_len, n_windows",
        [(np.zeros(shape=(1000,), dtype=float), 100, 10),
         (np.ones(shape=(1000,), dtype=float), 100, 10),
         (np.zeros(shape=(1000,), dtype=float), 100, 11),
         (np.ones(shape=(1000,), dtype=float), 100, 11),
         (np.zeros(shape=(1000,), dtype=float), 100, (6,7)),
         (np.ones(shape=(1000,), dtype=float), 100, (6,7)),
         (np.ones(shape=(1000,), dtype=float), 100, (6,0)),
         (np.zeros(shape=(1000,), dtype=float), 99, 10),
         (np.ones(shape=(1000,), dtype=float), 99, 10),]
)
def test_tailend_sequence_scan(dummy_data, seq_len, n_windows):
        strategy = TailendSequenceScanner(seq_len=seq_len, n_windows=n_windows)
        scanner = SequenceScannerContext(strategy=strategy)
        output_data = scanner(dummy_data)

        if isinstance(n_windows, int):
                assert len(output_data) == n_windows
        elif isinstance(n_windows, tuple):
                assert len(output_data) == sum(n_windows)

        for window in output_data:
                assert window.shape == (seq_len,)