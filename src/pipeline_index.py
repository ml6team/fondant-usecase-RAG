"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)


def create_pipeline(
    # fixed args
    pipeline_dir: str,
    embed_model_provider: str,
    embed_model: str,
    weaviate_url: str,
    weaviate_class_name: str,
    # custom args
    hf_dataset_name: str,
    data_column_name: str,
    n_rows_to_load: int,
    chunk_size: int,
    chunk_overlap: int,
):
    indexing_pipeline = Pipeline(
        pipeline_name="ingestion-pipeline",
        pipeline_description="Pipeline to prepare and process \
        data for building a RAG solution",
        base_path=pipeline_dir,  # The demo pipelines uses a local \
        # directory to store the data.
    )

    load_from_hf_hub = ComponentOp(
        component_dir="components/load_from_hf_hub",
        arguments={
            # Add arguments
            "dataset_name": hf_dataset_name,
            "column_name_mapping": {data_column_name: "text"},
            "n_rows_to_load": n_rows_to_load,
        },
        cache=False
    )

    chunk_text_op = ComponentOp.from_registry(
        name="chunk_text",
        arguments={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        cache=False
    )

    embed_text_op = ComponentOp.from_registry(
        name="embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
        },
        cache=False
    )

    index_weaviate_op = ComponentOp.from_registry(
        name="index_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class_name,
        },
        cache=False
    )

    # Construct your pipeline
    indexing_pipeline.add_op(load_from_hf_hub)
    indexing_pipeline.add_op(chunk_text_op, dependencies=load_from_hf_hub)
    indexing_pipeline.add_op(embed_text_op, dependencies=chunk_text_op)
    indexing_pipeline.add_op(index_weaviate_op, dependencies=embed_text_op)

    return indexing_pipeline


if __name__ == "__main__":
    pipeline = create_pipeline(
        pipeline_dir="./data-dir",
        embed_model_provider="huggingface",
        embed_model="all-MiniLM-L6-v2",
        weaviate_url="http://host.docker.internal:8080",
        weaviate_class_name="Pipeline_1",
        hf_dataset_name="wikitext@~parquet",
        data_column_name="text",
        n_rows_to_load=1000,
        chunk_size=512,
        chunk_overlap=32,
    )
