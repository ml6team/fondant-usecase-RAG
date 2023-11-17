import dask
import pandas as pd
from fondant.component import PandasTransformComponent
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import WeaviateVectorStore
from tqdm import tqdm

import weaviate

dask.config.set({"dataframe.convert-string": False})

class RetrieveChunks(PandasTransformComponent):
    def __init__(self, *_, weaviate_url: str, class_name: str, text_property_name: str, hf_embed_model: str, top_k: int) -> None:
        """
        Args:
            weaviate_url: An argument passed to the component
        """
        # Initialize your component here based on the arguments
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name
        self.text_property_name = text_property_name
        self.model = HuggingFaceEmbedding(hf_embed_model)
        self.k = top_k
        self.retriever = self._set_retriever(self.client, self.class_name, self.model, self.k)

    def _set_retriever(self, client, class_name, model, k):
        vector_store = WeaviateVectorStore(weaviate_client = client, index_name = class_name)
        service_context = ServiceContext.from_defaults(llm=None, embed_model=model)
        indexed_vector_db = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
        return indexed_vector_db.as_retriever(similarity_top_k=k)

    def retrieve_chunks(self, query: str):
        retrievals = self.retriever.retrieve(query)
        return [chunk.metadata[self.text_property_name] for chunk in retrievals]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "retrieved_chunks")] = dataframe[("text", "question")].apply(self.retrieve_chunks)
        return dataframe