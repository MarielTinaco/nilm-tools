
import numpy as np
import pytest

from src.data_processing.activation_scanning import *


def test_window_activation_scan_init():
        scanner = ActivationScannerContext()

        assert isinstance(scanner.strategy, WindowActivationScanner)


@pytest.mark.parametrize(
        "dummy_data, seq_len, n_windows",
        [(np.zeros(shape=(1000,), dtype=float), 100, 10),
         (np.ones(shape=(1000,), dtype=float), 100, 10),
         (np.zeros(shape=(100,), dtype=float), 100, 1),
         (np.ones(shape=(100,), dtype=float), 100, 1),
         (np.zeros(shape=(10,), dtype=float), 100, 1),
         (np.ones(shape=(10,), dtype=float), 100, 1),]
)
def test_window_activation_scan_shape(dummy_data, seq_len, n_windows):
        strategy = WindowActivationScanner(seq_len=seq_len, n_windows=n_windows)
        scanner = ActivationScannerContext(strategy=strategy)
        output_data = scanner(dummy_data)

        assert len(output_data) == n_windows

        for window in output_data:
                assert window.shape == (seq_len,)