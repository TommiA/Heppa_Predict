import pandas as pd
from typing import Dict, Iterable, List, Optional, Sequence, Union

DATA_FILE = "heppa_all_results.csv"
FEATURES = [
    "temperature",
    "horse_id",
    "track_condition",
    "location_id",
    "blooded",
]
GROUP_COLUMNS = [
    "location_id",
    "heat_num",
    "date_id",
]
CATEGORICAL_FEATURES = ["track_condition"]


def load_raw_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """Load the raw historical Heppa dataset from CSV."""
    return pd.read_csv(filepath)


def preprocess_heppa_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare raw Heppa race result data for model training."""
    df = raw_df.copy()

    df["placing"] = (
        pd.to_numeric(df["placing"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df = df[df["heat_num"] < 10].copy()

    df["m_heat_info"] = df["heat_info"].apply(
        lambda value: value if isinstance(value, str) and "ryhmä" in value else False
    )
    df = df[df["m_heat_info"] != False].copy()

    df["blooded"] = df["heat_info"].apply(
        lambda value: 0 if isinstance(value, str) and "Lämminveriset" in value else 1
    )

    df["horse_id"] = pd.factorize(df["horse"])[0]
    df["location_id"] = pd.factorize(df["location"])[0]
    df["date_id"] = pd.factorize(df["date"])[0]

    df["track_condition"] = df["track_condition"].astype("category")

    columns_to_keep = [
        "location",
        "date",
        "heat_num",
        "placing",
        "number",
        "horse",
        "temperature",
        "track_condition",
        "location_id",
        "date_id",
        "horse_id",
        "odds",
        "blooded",
        "heat_info",
    ]

    df = df[df["placing"] > 0].copy()
    return df[columns_to_keep].copy()


def add_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Create a ranking relevance signal from race placing values."""
    output = df.copy()
    output["relevance"] = output.groupby(GROUP_COLUMNS)["placing"].transform(
        lambda x: x.max() - x + 1
    )
    return output


def split_heat_groups(
    df: pd.DataFrame,
    test_size: float = 0.33,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique heats into training and validation sets by complete heat."""
    from sklearn.model_selection import train_test_split

    unique_heats = df[GROUP_COLUMNS].drop_duplicates().reset_index(drop=True)

    train_heats, val_heats = train_test_split(
        unique_heats,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df.merge(train_heats, on=GROUP_COLUMNS, how="inner")
    val_df = df.merge(val_heats, on=GROUP_COLUMNS, how="inner")

    return (
        train_df.sort_values(by=GROUP_COLUMNS).reset_index(drop=True),
        val_df.sort_values(by=GROUP_COLUMNS).reset_index(drop=True),
    )


def get_group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby(GROUP_COLUMNS).size().tolist()


def build_lgb_dataset(
    df: pd.DataFrame,
    label_column: str = "relevance",
    categorical_features: Optional[Sequence[str]] = None,
    reference: Optional["lgb.Dataset"] = None,
) -> "lgb.Dataset":
    import lightgbm as lgb

    dataset = lgb.Dataset(
        df[FEATURES],
        label=df[label_column],
        group=get_group_sizes(df),
        categorical_feature=list(categorical_features or CATEGORICAL_FEATURES),
        reference=reference,
    )
    return dataset


def build_value_map(values: pd.Series) -> Dict[str, int]:
    _, uniques = pd.factorize(values)
    return {entry: index for index, entry in enumerate(uniques)}


def map_values_to_id(
    values: Union[pd.Series, str],
    mapping: Dict[str, int],
    label: str,
) -> Union[pd.Series, Optional[int]]:
    if isinstance(values, pd.Series):
        mapped = values.map(mapping)
        missing = mapped.isna()
        if missing.any():
            missing_values = values[missing].dropna().unique().tolist()
            print(f"{label.capitalize()} values not found in training data: {missing_values}")
        return mapped

    mapped = mapping.get(values)
    if mapped is None:
        print(f"{label.capitalize()} '{values}' not found in training data")
    return mapped


def prepare_test_heat(
    horses: list[dict],
    location_name: str,
    heat_num: int,
    blooded: int,
    temperature: float,
    track_condition: str,
    location_to_id: Dict[str, int],
    horse_to_id: Dict[str, int],
    horse_counts: pd.Series,
    track_condition_categories: pd.Index,
) -> pd.DataFrame:
    test_heat = pd.DataFrame(horses).copy()

    test_heat["temperature"] = temperature
    test_heat["heat_num"] = heat_num
    test_heat["track_condition"] = track_condition
    test_heat["location"] = location_name
    test_heat["blooded"] = blooded

    test_heat["location_id"] = map_values_to_id(
        test_heat["location"],
        location_to_id,
        label="location",
    )

    test_heat["horse_id"] = map_values_to_id(
        test_heat["horse"],
        horse_to_id,
        label="horse",
    )

    test_heat["occurrences"] = (
        test_heat["horse"]
        .map(horse_counts)
        .fillna(0)
        .astype(int)
    )

    test_heat["track_condition"] = pd.Categorical(
        test_heat["track_condition"],
        categories=track_condition_categories,
    )

    return test_heat


def predict_heat(
    model: "lgb.Booster",
    test_heat: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    test_heat = test_heat.copy()
    test_heat["pred_score"] = model.predict(test_heat[features or FEATURES])
    test_heat = test_heat.sort_values(by="pred_score", ascending=False).reset_index(drop=True)
    test_heat["predicted_placing"] = test_heat.index + 1
    return test_heat
