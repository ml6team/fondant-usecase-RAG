"""Pipeline to prepare and process data for building a RAG solution using LlamaIndex"""
import logging
from pathlib import Path

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)

BASE_PATH = "./data-dir"
BASE_PATH = create_directory_if_not_exists(BASE_PATH)

pipeline = Pipeline(
    pipeline_name="ingestion-pipeline-li",  
    pipeline_description="Pipeline to prepare and process \
    data for building a RAG solution using LlamaIndex",
    base_path=BASE_PATH,  # The demo pipelines uses a local \
    # directory to store the data.
)

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        # Add arguments
        "dataset_name": "BeIR/webis-touche2020@~parquet",
        "column_name_mapping": {"text": "text_data"},
        "n_rows_to_load": 10,
    },
)

text_cleaning = ComponentOp(
    component_dir="components/text_cleaning"
)

get_llama_index_nodes = ComponentOp(
    component_dir="components/li_create_nodes",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    cache=False
)

embed_text_op = ComponentOp(
    component_dir="components/li_get_embeddings",
    arguments={"hf_embed_model": "BAAI/bge-small-en"},
    # number_of_gpus=1,
)

index_weaviate_op = ComponentOp(
    component_dir="components/li_write_to_vector_db",
    arguments={
        "local_url": "http://host.docker.internal:8080",
        "index_name": "Llama_Paper"
    },
)

# Construct your pipeline
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(get_llama_index_nodes, dependencies=load_from_hf_hub)
pipeline.add_op(embed_text_op, dependencies=get_llama_index_nodes)
pipeline.add_op(index_weaviate_op, dependencies=embed_text_op)
