import dask.dataframe as dd
import pyarrow as pa
from fondant.component import DaskTransformComponent
from fondant.pipeline import lightweight_component


@lightweight_component(
    produces={
        "metric": pa.string(),
        "score": pa.float32()
    }
)
class AggregateResults(DaskTransformComponent):
    def __init__(self, consumes: dict, **kwargs):
        self.consumes = consumes

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        metrics = list(self.consumes.keys())
        agg = dataframe[metrics].mean()
        agg_df = agg.to_frame(name="score")
        agg_df["metric"] = agg.index
        agg_df.index = agg_df.index.astype(str)

        return agg_df
