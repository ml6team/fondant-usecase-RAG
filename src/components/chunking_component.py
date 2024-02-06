import typing as t

import pandas as pd
import pyarrow as pa
from fondant.component import PandasTransformComponent
from fondant.pipeline import lightweight_component


@lightweight_component(
    consumes={"text": pa.string()},
    produces={"text": pa.string(), "original_document_id": pa.string()},
    extra_requires=["langchain==0.0.329"],
)
class ChunkTextComponent(PandasTransformComponent):
    """Component that chunks text into smaller segments.
    More information about the different chunking strategies can be here:
      - https://python.langchain.com/docs/modules/data_connection/document_transformers/
      - https://www.pinecone.io/learn/chunking-strategies/.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """
        Args:
            chunk_size: the chunk size
            chunk_overlap: the overlap between chunks.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(self, row) -> t.List[t.Tuple]:
        # Multi-index df has id under the name attribute
        doc_id = row.name
        text_data = row["text"]
        docs = self.chunker.create_documents([text_data])

        return [
            (doc_id, f"{doc_id}_{chunk_id}", chunk.page_content)
            for chunk_id, chunk in enumerate(docs)
        ]

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        import itertools

        results = dataframe.apply(
            self.chunk_text,
            axis=1,
        ).to_list()

        # Flatten results
        results = list(itertools.chain.from_iterable(results))

        # Turn into dataframes
        results_df = pd.DataFrame(
            results,
            columns=["original_document_id", "id", "text"],
        )
        results_df = results_df.set_index("id")

        return results_df
