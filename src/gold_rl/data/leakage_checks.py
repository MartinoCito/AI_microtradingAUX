from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
from pandas.testing import assert_frame_equal


FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def assert_future_append_invariant(
    feature_builder: FeatureBuilder,
    history_dataframe: pd.DataFrame,
    future_dataframe: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    past_features = feature_builder(history_dataframe)

    appended_dataframe = pd.concat(
        [history_dataframe, future_dataframe],
        axis=0,
    ).sort_index(kind="stable")

    appended_features = feature_builder(appended_dataframe)
    compare_columns = list(feature_columns or past_features.columns)

    assert_frame_equal(
        past_features.loc[:, compare_columns],
        appended_features.loc[past_features.index, compare_columns],
        check_dtype=False,
        check_exact=False,
        rtol=rtol,
        atol=atol,
    )


def assert_past_normalization_invariant(
    past_features: pd.DataFrame,
    appended_features: pd.DataFrame,
    normalized_past: pd.DataFrame,
    normalized_appended: pd.DataFrame,
    feature_columns: Sequence[str],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    assert_frame_equal(
        past_features.loc[:, feature_columns],
        appended_features.loc[past_features.index, feature_columns],
        check_dtype=False,
        check_exact=False,
        rtol=rtol,
        atol=atol,
    )

    assert_frame_equal(
        normalized_past.loc[:, feature_columns],
        normalized_appended.loc[past_features.index, feature_columns],
        check_dtype=False,
        check_exact=False,
        rtol=rtol,
        atol=atol,
    )
