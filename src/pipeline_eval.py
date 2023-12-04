"""Pipeline used to evaluate a RAG pipeline."""
import logging

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)


pipeline = Pipeline(
    pipeline_name="evaluation-pipeline",
    pipeline_description="Pipeline to evaluate \
    a RAG solution",
    base_path="./data-dir",  # The demo pipelines uses a local \
    # directory to store the data.
)

load_from_csv = ComponentOp(
    component_dir="components/load_from_csv",
    arguments={
        # Add arguments
        "dataset_uri": "/data/wikitext_1000_q.csv",
        "column_separator": ";",
        "column_name_mapping": {"question": "text"},
    },
)

embed_text_op = ComponentOp.from_registry(
    name="embed_text",
    arguments={
        "model_provider": "huggingface",
        "model": "all-MiniLM-L6-v2",
    },
)

retrieve_chunks = ComponentOp(
    component_dir="components/retrieve_from_weaviate",
    arguments={
        "weaviate_url": "http://host.docker.internal:8080",
        "class_name": "Index",
        "top_k": 2,
    },
)

retriever_eval = ComponentOp(
    component_dir="components/retriever_eval",
    arguments={
        "llm_name": "OpenAI",
        "llm_kwargs": {"openai_api_key": ""},
        "metrics": ["context_precision", "context_relevancy"],
    },
)

aggregate_results = ComponentOp(
    component_dir="components/aggregate_eval_results",
)

# Construct your pipeline
pipeline.add_op(load_from_csv)
pipeline.add_op(embed_text_op, dependencies=load_from_csv)
pipeline.add_op(retrieve_chunks, dependencies=embed_text_op)
pipeline.add_op(retriever_eval, dependencies=retrieve_chunks)
pipeline.add_op(aggregate_results, dependencies=retriever_eval)
