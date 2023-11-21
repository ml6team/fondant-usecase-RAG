import dask
import pandas as pd
from fondant.component import PandasTransformComponent

import weaviate


class RetrieveChunks(PandasTransformComponent):
    def __init__(
            self,
            *_,
            weaviate_url: str,
            class_name: str,
            top_k: int
        ) -> None:
        """
        Args:
            weaviate_url: An argument passed to the component
        """
        # Initialize your component here based on the arguments
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name
        self.k = top_k
    
    def retrieve_chunks(self, vector_query: str):
        """Get results from weaviate database"""
    
        result = (
        self.client.query
        .get(self.class_name, ["passage"])
        .with_near_vector({"vector":vector_query})
        .with_limit(self.k)
        .with_additional(["distance"])
        .do()
        )
        result_dict = result["data"]["Get"][self.class_name]
        text = [retrieved_chunk["passage"] for retrieved_chunk in result_dict]

        return text

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "retrieved_chunks")] = dataframe[("text", "embedding")].apply(
            self.retrieve_chunks
        )
        return dataframe
