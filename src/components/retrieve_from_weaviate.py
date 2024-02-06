import typing as t
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from fondant.component import PandasTransformComponent
from fondant.pipeline import lightweight_component

@lightweight_component(
    produces={"retrieved_chunks": pa.list_(pa.string())},
    extra_requires=["weaviate-client==3.24.1"]
)
class RetrieveFromWeaviateComponent(PandasTransformComponent):
    def __init__(
        self,
        *,
        weaviate_url: str,
        class_name: str,
        top_k: int,
    ) -> None:
        """
        Args:
            weaviate_url: An argument passed to the component.
            class_name: Name of class to query
            top_k: Amount of context to return.
            additional_config: Additional configuration passed to the weaviate client.
            additional_headers: Additional headers passed to the weaviate client.
            hybrid_query: The hybrid query to be used for retrieval. Optional parameter.
            hybrid_alpha: Argument to change how much each search affects the results. An alpha
             of 1 is a pure vector search. An alpha of 0 is a pure keyword search.
            rerank: Whether to rerank the results based on the hybrid query. Defaults to False.
             Check this notebook for more information on reranking:
             https://github.com/weaviate/recipes/blob/main/ranking/cohere-ranking/cohere-ranking.ipynb
             https://weaviate.io/developers/weaviate/search/rerank.
        """
        import weaviate

        # Initialize your component here based on the arguments
        self.client = weaviate.Client(
            url=weaviate_url,
            additional_config=None,
            additional_headers=None,
        )
        self.class_name = class_name
        self.k = top_k

    def teardown(self) -> None:
        del self.client

    def retrieve_chunks_from_embeddings(self, vector_query: list):
        """Get results from weaviate database."""
        query = (
            self.client.query.get(self.class_name, ["passage"])
            .with_near_vector({"vector": vector_query})
            .with_limit(self.k)
            .with_additional(["distance"])
        )

        result = query.do()
        if "data" in result:
            result_dict = result["data"]["Get"][self.class_name]
            return [retrieved_chunk["passage"] for retrieved_chunk in result_dict]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        if "embedding" in dataframe.columns:
            dataframe["retrieved_chunks"] = dataframe["embedding"].apply(
                self.retrieve_chunks_from_embeddings,
            )

        elif "prompt" in dataframe.columns:
            dataframe["retrieved_chunks"] = dataframe["prompt"].apply(
                self.retrieve_chunks_from_prompts,
            )
        else:
            msg = "Dataframe must contain either an 'embedding' column or a 'prompt' column."
            raise ValueError(
                msg,
            )

        return dataframe