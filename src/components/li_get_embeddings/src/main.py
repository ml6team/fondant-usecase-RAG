import dask
import pandas as pd
from fondant.component import PandasTransformComponent
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import TextNode
from tqdm import tqdm

dask.config.set({"dataframe.convert-string": False})

class GenerateEmbeddings(PandasTransformComponent):
    def __init__(self, *_, hf_embed_model: str) -> None:
        """
        Args:
            argumentX: An argument passed to the component
        """
        # Initialize your component here based on the arguments
        self.model = HuggingFaceEmbedding(hf_embed_model)

    def deserialise_nodes(self, node: str):
        return TextNode.from_json(node)
    
    def get_embeddings(self, row):
        node = self.deserialise_nodes(row)
        node_embedding = self.model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
        return node.to_json()

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe[("text", "node")] = dataframe[("text", "node")].apply(self.get_embeddings)
        return dataframe