"""Fondant pipeline to evaluate a RAG pipeline."""

import pyarrow as pa
from fondant.pipeline import Pipeline, Resources


def create_pipeline(
    *,
    base_path: str = "./data",
    weaviate_url="http://host.docker.internal:8080",
    weaviate_class: str = "Pipeline1",
    evaluation_set_path="./evaluation_datasets",
    evaluation_set_filename="wikitext_1000_q.csv",
    evaluation_set_separator: str = ";",
    embed_model_provider: str = "huggingface",
    embed_model: str = "all-MiniLM-L6-v2",
    embed_api_key: dict = {},
    retrieval_top_k: int = 3,
    evaluation_module: str = "langchain.llms",
    evaluation_llm: str = "OpenAI",
    evaluation_llm_kwargs: dict = {"model_name": "gpt-3.5-turbo"},
    evaluation_metrics: list = ["context_precision", "context_relevancy"],
    number_of_accelerators=None,
    accelerator_name=None,
):
    """Create a Fondant pipeline based on the provided arguments."""
    evaluation_pipeline = Pipeline(
        name="evaluation-pipeline",
        description="Pipeline to evaluate a RAG solution",
        base_path=base_path,
    )

    load_from_csv = evaluation_pipeline.read(
        "load_from_csv",
        arguments={
            "dataset_uri": "/evaldata/" + evaluation_set_filename,
            # mounted dir from within docker as extra_volumes
            "column_separator": evaluation_set_separator,
        },
        produces={
            "question": pa.string(),
        },
    )

    embed_text_op = load_from_csv.apply(
        "embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
            "api_keys": embed_api_key,
        },
        consumes={
            "text": "question",
        },
        resources=Resources(
            accelerator_number=number_of_accelerators,
            accelerator_name=accelerator_name,
        ),
        cluster_type="local" if number_of_accelerators is not None else "default",
    )

    retrieve_chunks = embed_text_op.apply(
        "retrieve_from_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class,
            "top_k": retrieval_top_k,
        },
        cache=False,
    )

    retriever_eval = retrieve_chunks.apply(
        "evaluate_ragas",
        arguments={
            "module": evaluation_module,
            "llm_name": evaluation_llm,
            "llm_kwargs": evaluation_llm_kwargs,
        },
        produces={metric: pa.float32() for metric in evaluation_metrics},
    )

    retriever_eval.apply(
        "components/aggregate_eval_results",
        consumes={metric: pa.float32() for metric in evaluation_metrics},
    )

    return evaluation_pipeline
