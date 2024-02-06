"""Fondant pipeline to index a RAG system."""
import pyarrow as pa
from fondant.pipeline import Pipeline, Resources
from pathlib import Path
from components.chunking_component import ChunkTextComponent

def create_pipeline(
    *,
    weaviate_url: str,
    base_path: str = "./data",
    n_rows_to_load: int = 1000,
    weaviate_class: str = "Pipeline1",
    embed_model_provider: str = "huggingface",
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_args: dict = {"chunk_size": 512, "chunk_overlap": 32},
    number_of_accelerators=None,
    accelerator_name=None,
):
    """Create a Fondant pipeline based on the provided arguments."""


    Path(base_path).mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        name="indexing-pipeline",
        description="Pipeline to prepare and process data for building a RAG solution",
        base_path=base_path
        )

    text = pipeline.read(
        "load_from_hf_hub",
        arguments={
            # Add arguments
            "dataset_name": "wikitext@~parquet",
            "n_rows_to_load": n_rows_to_load,
        },
        produces={
            "text": pa.string()
        }
    )


    chunks = text.apply(
        ChunkTextComponent,
        arguments=chunk_args
    )


    embeddings = chunks.apply(
        "embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model
        },
        resources=Resources(
            accelerator_number=number_of_accelerators,
            accelerator_name=accelerator_name,
        ),
        cluster_type="local" if number_of_accelerators is not None else "default",
        cache=False
    )

    embeddings.write(
        "index_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class,
        },
        consumes={
            "text": pa.string(),
            "embedding": pa.list_(pa.float32()),   
        }
    )

    return pipeline
