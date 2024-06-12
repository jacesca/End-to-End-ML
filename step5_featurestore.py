import pandas as pd

from environment import PARQUET_FILE_PATH, FEATURE_STORE_PATH
from feast import Field, Entity, FeatureStore, FileSource, FeatureView
from feast.types import Float32, Int32
from datetime import datetime


def store_feature(X: pd.DataFrame) -> None:
    X_feature = X.reset_index(names='patient_id')
    X_feature['timestamp'] = datetime.now()

    # Define the entity, which in this case is a patient, and features
    patient = Entity(name='patient', join_keys=['patient_id'])
    cp = Field(name='cp', dtype=Float32)
    thalach = Field(name='thalach', dtype=Int32)
    ca = Field(name='ca', dtype=Int32)
    thal = Field(name='thal', dtype=Int32)

    # Saving data in parquet format (just to try a different format)
    X_feature.to_parquet(PARQUET_FILE_PATH)

    # Point File Source to the saved file
    data_source = FileSource(
        path=PARQUET_FILE_PATH,
        event_timestamp_column='timestamp',
        created_timestamp_column='created'
    )

    # Create a Feature View of the features
    heart_disease_fv = FeatureView(
        name='heart_disease',
        entities=[patient],
        schema=[cp, thalach, ca, thal],
        source=data_source
    )

    # Create a store of the data and apply the features
    store = FeatureStore(repo_path=FEATURE_STORE_PATH)
    store.apply([patient, heart_disease_fv])
