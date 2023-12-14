"""Fondant pipeline to index a RAG system."""
import pyarrow as pa
from fondant.pipeline import Pipeline


def create_pipeline(
    *,
    base_path: str = "s3://sagemaker-fondant-artifacts-robbe/data",
    n_rows_to_load: int = 1000,
    weaviate_url: str = "http://host.docker.internal:8080",
    weaviate_class: str = "Pipeline1",
    weaviate_overwrite: bool = True,
    embed_model_provider: str = "huggingface",
    embed_model: str = "all-MiniLM-L6-v2",
    embed_api_key: dict = {},
    chunk_size: int = 512,
    chunk_overlap: int = 32,
):
    """Create a Fondant pipeline based on the provided arguments."""
    indexing_pipeline = Pipeline(
        name="indexing-pipeline",
        description="Pipeline to prepare and process data for building a RAG solution",
        base_path=base_path,
    )

    text = indexing_pipeline.read(
        "load_from_hf_hub",
        arguments={
            # Add arguments
            "dataset_name": "wikitext@~parquet",
            "n_rows_to_load": n_rows_to_load,
        },
        produces={
            "text": pa.string(),
        },
    )

    chunks = text.apply(
        "chunk_text",
        arguments={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    embeddings = chunks.apply(
        "embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
            "api_keys": embed_api_key,
        },
    )

    embeddings.write(
        "index_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class,
            "overwrite": weaviate_overwrite,
        },
        cache=False,
    )

    return indexing_pipeline
