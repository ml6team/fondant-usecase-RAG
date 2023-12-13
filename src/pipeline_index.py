"""Fondant pipeline to index a RAG system."""
import pyarrow as pa
from fondant.pipeline import Pipeline


def create_pipeline(  # noqa: PLR0913
    pipeline_dir: str = "./data-dir",
    embed_model_provider: str = "huggingface",
    embed_model: str = "all-MiniLM-L6-v2",
    api_keys: dict = {},
    weaviate_url: str = "http://host.docker.internal:8080",
    weaviate_class_name: str = "Pipeline1",
    overwrite: bool = True,
    # indexing args
    hf_dataset_name: str = "wikitext@~parquet",
    data_column_name: str = "text",
    n_rows_to_load: int = 1000,
    chunk_size: int = 512,
    chunk_overlap: int = 32,
):
    indexing_pipeline = Pipeline(
        name="indexing-pipeline",
        description="Pipeline to prepare and process data for building a RAG solution",
        base_path=pipeline_dir,  # The demo pipelines uses a local directory to store the data.
    )

    text = indexing_pipeline.read(
        "load_from_hf_hub",
        arguments={
            # Add arguments
            "dataset_name": hf_dataset_name,
            "column_name_mapping": {data_column_name: "text"},
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
            "api_keys": api_keys,
        },
    )

    embeddings.write(
        "index_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class_name,
            "overwrite": overwrite,
        },
    )

    return indexing_pipeline
