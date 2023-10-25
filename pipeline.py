"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
from pathlib import Path
from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)


pipeline = Pipeline(
    pipeline_name="ingestion-pipeline",  # Add a unique pipeline name to easily track your progress and data
    pipeline_description="Pipeline to prepare and process data for building a RAG solution",
    base_path=create_directory_if_not_exists("./data-dir"), # The demo pipelines uses a local directory to store the data.
)

load_from_parquet = ComponentOp(
    component_dir="components/load_from_parquet",
    arguments={
        # Add arguments
        "dataset_uri": "https://huggingface.co/datasets/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/test/0000.parquet",
        "column_name_mapping": {
            "text": "text_data"
        },
        "n_rows_to_load": 10
    }
)

chunk_text_op = ComponentOp.from_registry(
    name="chunk_text",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 32,
    },
    input_partition_rows=10,
)

embed_text_op = ComponentOp.from_registry(
    name="embed_text",
    arguments={
        "model_provider": "huggingface",
        "model": "all-MiniLM-L6-v2",
    },
)

index_weaviate_op = ComponentOp.from_registry(
    name="index_weaviate",
    arguments={
        "weaviate_url": "http://host.docker.internal:8080",
        "class_name": "index",  # Add a unique class name to show up on the leaderboard
    },
)

# Construct your pipeline
pipeline.add_op(load_from_parquet)
pipeline.add_op(chunk_text_op, dependencies=load_from_parquet)
pipeline.add_op(embed_text_op, dependencies=chunk_text_op)
pipeline.add_op(index_weaviate_op, dependencies=embed_text_op)
