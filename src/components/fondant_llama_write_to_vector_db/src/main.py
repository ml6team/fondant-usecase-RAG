from typing import Optional

import dask
import dask.dataframe as dd
from fondant.component import DaskWriteComponent
from llama_index.schema import TextNode
from llama_index.vector_stores import WeaviateVectorStore

import weaviate

# from fondant.component_spec import ComponentSpec

dask.config.set({"dataframe.convert-string": False})

class WriteToVectorDB(DaskWriteComponent):
    def __init__(
            self,
            *_,
            cloud_store: bool,
            wcs_username: Optional[str],
            wcs_password: Optional[str],
            wcs_url: Optional[str],
            local_url: Optional[str],
            index_name: str
            ):
        if cloud_store:
            resource_owner_config = weaviate.AuthClientPassword(
                username=wcs_username,
                password=wcs_password,
            )
            self.client = weaviate.Client(url=wcs_url, auth_client_secret=resource_owner_config)
        else:
            self.client = weaviate.Client(url=local_url)
        self.vector_store = WeaviateVectorStore(weaviate_client=self.client, index_name=index_name)

    @staticmethod
    def create_nodes(row: str):
        return TextNode(
            text=row["text_data"],
            embedding=row["text_embedding"].tolist()
        )
    
    def load_partitions(self, df, vector_store):
        nodes = df.apply(self.create_nodes, axis=1)
        vector_store.add(nodes)

    def write(self, dataframe: dd.DataFrame):
        vector_store = self.vector_store

        dataframe.map_partitions(
            self.load_partitions,
            vector_store=vector_store,
            meta=("nodes", "object"),
        ).compute()