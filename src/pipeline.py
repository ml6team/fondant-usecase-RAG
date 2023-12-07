"""Fondant pipeline to index a RAG system."""
import pyarrow as pa
from fondant.pipeline import Pipeline

pipeline = Pipeline(
    name="ingestion-pipeline",
    description="Pipeline to prepare and process data for building a RAG solution",
    base_path="./data-dir",  # The demo pipelines uses a local directory to store the data.
)

text = pipeline.read(
    "load_from_hf_hub",
    arguments={
        # Add arguments
        "dataset_name": "wikitext@~parquet",
        "column_name_mapping": {"text": "text"},
        "n_rows_to_load": 1000,
    },
    produces={
        "text": pa.string(),
    },
)

chunks = text.apply(
    "chunk_text",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 32,
    },
)

embeddings = chunks.apply(
    "embed_text",
    arguments={
        "model_provider": "huggingface",
        "model": "all-MiniLM-L6-v2",
    },
)

embeddings.write(
    "index_weaviate",
    arguments={
        "weaviate_url": "http://host.docker.internal:8080",
        "class_name": "index",
    },
)
