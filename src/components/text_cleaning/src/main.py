import logging
import pandas as pd
from fondant.component import PandasTransformComponent


logger = logging.getLogger(__name__)


class TextCleaningComponent(PandasTransformComponent):  
    def __init__(self, *_):
        """Initialize your component"""

    def remove_empty_lines(self, text):
        lines = text.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(non_empty_lines)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "data")] = dataframe[("text", "data")].apply(lambda x: self.remove_empty_lines)
        return dataframe
