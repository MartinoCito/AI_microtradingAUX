import pandas as pd
import pytest

from gold_rl.data.resampling import (
    resample_ohlcv,
    shift_open_timestamps_to_close,
    validate_resampled_ohlcv,
)


def build_m1_sample() -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 10:00",
        periods=30,
        freq="1min",
    )

    data = {
        "Open": range(30),
        "High": [value + 1 for value in range(30)],
        "Low": [value - 1 for value in range(30)],
        "Close": range(30),
        "Volume": [1.0] * 30,
    }

    return pd.DataFrame(data, index=index)


def test_shift_moves_timestamps_forward():
    m1 = build_m1_sample()
    shifted = shift_open_timestamps_to_close(m1, bar_duration="1min")

    assert shifted.index[0] == m1.index[0] + pd.Timedelta(minutes=1)


def test_resample_uses_only_past_data():
    m1 = build_m1_sample()
    shifted = shift_open_timestamps_to_close(m1, bar_duration="1min")

    m15 = resample_ohlcv(
        shifted,
        target_rule="15min",
        min_coverage_ratio=1.0,
        source_bar_duration="1min",
    )

    first_bar_label = m15.index[0]
    contributing_m1 = shifted.loc[shifted.index <= first_bar_label]

    assert contributing_m1.index.max() == first_bar_label
    assert (shifted.loc[shifted.index > first_bar_label].index
            .isin(contributing_m1.index)).sum() == 0


def test_incomplete_window_is_dropped():
    m1 = build_m1_sample().iloc[:10]
    shifted = shift_open_timestamps_to_close(m1, bar_duration="1min")

    m15 = resample_ohlcv(
        shifted,
        target_rule="15min",
        min_coverage_ratio=1.0,
        source_bar_duration="1min",
    )

    assert m15.empty


def test_validate_resampled_ohlcv_rejects_invalid_bars():
    invalid = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [9.0],
            "Low": [8.0],
            "Close": [10.0],
            "Volume": [1.0],
            "bar_count": [15],
            "expected_bar_count": [15],
        },
        index=pd.date_range("2024-01-01 10:15", periods=1, freq="15min"),
    )

    with pytest.raises(ValueError):
        validate_resampled_ohlcv(invalid)