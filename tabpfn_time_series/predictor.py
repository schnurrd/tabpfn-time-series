import logging
from enum import Enum
import numpy as np

from autogluon.timeseries import TimeSeriesDataFrame

from tabpfn_time_series.tabpfn_worker import TabPFNClient, LocalTabPFN, MockTabPFN
from tabpfn_time_series.defaults import (
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    TABPFN_TS_DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)


class TabPFNMode(Enum):
    LOCAL = "tabpfn-local"
    CLIENT = "tabpfn-client"
    MOCK = "tabpfn-mock"

def convert_to_differences(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    def calculate_diff(group):
        group["target_diff"] = group["target"].diff().fillna(0)
        return group

    tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(calculate_diff)
    return tsdf.drop(columns=["target"]).rename(columns={"target_diff": "target"})

def postprocess_predictions(pred_tsdf: TimeSeriesDataFrame, last_target_values: dict) -> TimeSeriesDataFrame:
    def add_last_target_value(group):
        item_id = group.index.get_level_values("item_id")[0]
        group["target"] = pred_tsdf["target"].cumsum() + last_target_values[item_id]
        return group

    pred_tsdf = pred_tsdf.groupby(level="item_id", group_keys=False).apply(add_last_target_value)
    return pred_tsdf


class TabPFNTimeSeriesPredictor:
    """
    Given a TimeSeriesDataFrame (multiple time series), perform prediction on each time series individually.
    """

    def __init__(
        self,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        config: dict = TABPFN_TS_DEFAULT_CONFIG,
    ) -> None:
        worker_mapping = {
            TabPFNMode.CLIENT: lambda: TabPFNClient(config),
            TabPFNMode.LOCAL: lambda: LocalTabPFN(config),
            TabPFNMode.MOCK: lambda: MockTabPFN(config),
        }
        self.tabpfn_worker = worker_mapping[tabpfn_mode]()

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,  # with features and target
        test_tsdf: TimeSeriesDataFrame,  # with features only
        quantile_config: list[float] = TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    ) -> TimeSeriesDataFrame:
        """
        Predict on each time series individually (local forecasting).
        """
        self.last_target_values = {}
        # Store the last target value for each item_id
        for item_id in train_tsdf.item_ids:
            self.last_target_values[item_id] = train_tsdf.xs(item_id, level="item_id")["target"].iloc[-1]

        # Convert training data to differences
        train_tsdf = convert_to_differences(train_tsdf)

        logger.info(
            f"Predicting {len(train_tsdf.item_ids)} time series with config{self.tabpfn_worker.config}"
        )
        
        # Generate predictions
        pred_tsdf = self.tabpfn_worker.predict(train_tsdf, test_tsdf, quantile_config)

        print("Got TS Back")
        
        # Postprocess predictions
        pred_tsdf = postprocess_predictions(pred_tsdf, self.last_target_values)

        return pred_tsdf
