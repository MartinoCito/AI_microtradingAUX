import pandas as pd


def chronological_date_split(
    dataframe: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide il dataset cronologicamente senza mescolare i dati.
    """
    if dataframe.empty:
        raise ValueError("Il dataset è vuoto.")

    train_end_timestamp = pd.Timestamp(train_end)
    validation_end_timestamp = pd.Timestamp(validation_end)

    if train_end_timestamp >= validation_end_timestamp:
        raise ValueError(
            "train_end deve precedere validation_end."
        )

    train = dataframe.loc[
        dataframe.index <= train_end_timestamp
    ].copy()

    validation = dataframe.loc[
        (dataframe.index > train_end_timestamp)
        & (dataframe.index <= validation_end_timestamp)
    ].copy()

    test = dataframe.loc[
        dataframe.index > validation_end_timestamp
    ].copy()

    if train.empty:
        raise ValueError("Il training set è vuoto.")

    if validation.empty:
        raise ValueError("Il validation set è vuoto.")

    if test.empty:
        raise ValueError("Il test set è vuoto.")

    if train.index.max() >= validation.index.min():
        raise RuntimeError(
            "Train e validation si sovrappongono."
        )

    if validation.index.max() >= test.index.min():
        raise RuntimeError(
            "Validation e test si sovrappongono."
        )

    return train, validation, tests