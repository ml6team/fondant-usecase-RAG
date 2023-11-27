import dask.dataframe as dd
from fondant.component import DaskTransformComponent


class AggregateResults(DaskTransformComponent):
    def __init__(self, *_, metrics: list) -> None:
        """
        Args:
            metrics: Metrics chosen to be aggregated
        """

        self.metrics = metrics

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        if self.metrics is None:
            chosen_metrics = list(
                dataframe["text"].select_dtypes(["float", "int"]).columns
            )
        else:
            chosen_metrics = self.metrics
        metrics = [f"text_{metric}" for metric in chosen_metrics]

        agg = dataframe[metrics].mean()
        agg_df = agg.to_frame(name="score")
        agg_df["metric"] = agg.index
        agg_results_df = agg_df[["metric", "score"]]
        agg_results_df = agg_results_df.reset_index(drop=True).rename(
            columns={"metric": "text_metric", "score": "text_score"}
        )

        return agg_results_df
