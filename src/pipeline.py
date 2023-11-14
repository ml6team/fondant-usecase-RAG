"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)


pipeline = Pipeline(
    pipeline_name="ingestion-pipeline",  
    pipeline_description="Pipeline to prepare and process \
    data for building a RAG solution",
    base_path="./data-dir",  # The demo pipelines uses a local \
    # directory to store the data.
)

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        # Add arguments
        "dataset_name": "wikitext@~parquet",
        "column_name_mapping": {"text": "text_data"},
        "n_rows_to_load": 10,
    },
)

chunk_text_op = ComponentOp.from_registry(
    name="chunk_text",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 32,
    },
)

embed_text_op = ComponentOp.from_registry(
    name="embed_text",
    arguments={
        "model_provider": "huggingface",
        "model": "all-MiniLM-L6-v2",
    },
)

# index_weaviate_op = ComponentOp.from_registry(
#     name="index_weaviate",
#     arguments={
#         "weaviate_url": "http://host.docker.internal:8080",
#         "class_name": "index"
#     },
# )

#from fondant to llama_index --> create and fill a vector store with Nodes
index_weaviate_op = ComponentOp(
    component_dir="components/fondant_llama_write_to_vector_db",
    arguments={
        "local_url": "http://host.docker.internal:8080",
        "index_name": "Llama_Paper"
    },
)

# Construct your pipeline
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(chunk_text_op, dependencies=load_from_hf_hub)
pipeline.add_op(embed_text_op, dependencies=chunk_text_op)
pipeline.add_op(index_weaviate_op, dependencies=embed_text_op)
