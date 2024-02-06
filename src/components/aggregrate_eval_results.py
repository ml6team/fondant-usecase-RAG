import dask.dataframe as dd
import pyarrow as pa
from fondant.component import DaskTransformComponent
from fondant.pipeline import lightweight_component


@lightweight_component(
    consumes={
    "context_precision": pa.float32(),
    "context_relevancy": pa.float32(),
    },
    produces={
        "metric": pa.string(),
        "score": pa.float32()
    }
)
class AggregateResults(DaskTransformComponent):

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        metrics = list(self.consumes.keys())
        agg = dataframe[metrics].mean()
        agg_df = agg.to_frame(name="score")
        agg_df["metric"] = agg.index
        agg_df.index = agg_df.index.astype(str)

        return agg_df
