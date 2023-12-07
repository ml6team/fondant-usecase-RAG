import pandas as pd
from fondant.component import PandasTransformComponent


class TextCleaningComponent(PandasTransformComponent):
    def __init__(self, **kwargs):
        """Initialize your component."""

    def remove_empty_lines(self, text):
        lines = text.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(non_empty_lines)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["text"] = dataframe["text"].apply(
            self.remove_empty_lines,
        )
        return dataframe
