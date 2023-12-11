"""Fondant pipeline to evaluate a RAG pipeline."""
from fondant.pipeline import Pipeline


def create_pipeline(  # noqa: PLR0913
    pipeline_dir: str = "./data-dir",
    embed_model_provider: str = "huggingface",
    embed_model: str = "all-MiniLM-L6-v2",
    weaviate_url="http://host.docker.internal:8080",
    weaviate_class_name: str = "Pipeline1",
    # evaluation args
    csv_dataset_uri: str = "/data/wikitext_1000_q.csv",
    csv_column_separator: str = ";",
    question_column_name: str = "question",
    top_k: int = 3,
    module: str = "langchain.llms",
    llm_name: str = "OpenAI",
    llm_kwargs: dict = {"openai_api_key": ""},  # TODO if use Fondant CLI
    metrics: list = ["context_precision", "context_relevancy"],
):
    evaluation_pipeline = Pipeline(
        name="evaluation-pipeline",
        description="Pipeline to evaluate \
        a RAG solution",
        base_path=pipeline_dir,  # The demo pipelines uses a local \
        # directory to store the data.
    )

    load_from_csv = evaluation_pipeline.read(
        "components/load_from_csv",
        arguments={
            # Add arguments
            "dataset_uri": csv_dataset_uri,
            "column_separator": csv_column_separator,
            "column_name_mapping": {question_column_name: "text"},
        },
    )

    embed_text_op = load_from_csv.apply(
        "embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
        },
    )

    retrieve_chunks = embed_text_op.apply(
        "components/retrieve_from_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class_name,
            "top_k": top_k,
        },
    )

    retriever_eval = retrieve_chunks.apply(
        "components/retriever_eval",
        arguments={
            "module": module,
            "llm_name": llm_name,
            "llm_kwargs": llm_kwargs,
            "metrics": metrics,
        },
    )

    retriever_eval.apply(
        "components/aggregate_eval_results",
    )

    return evaluation_pipeline