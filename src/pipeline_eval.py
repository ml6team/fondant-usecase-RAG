"""Pipeline used to evaluate a RAG pipeline."""
import logging

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

def create_pipeline(
        #fixed args
        pipeline_dir: str,
        embed_model_provider: str,
        embed_model: str,
        weaviate_url: str,
        weaviate_class_name: str,
        #custom args
        csv_dataset_uri: str,
        csv_column_separator: str,
        question_column_name: str,
        top_k: int,
        llm_name: str,
        llm_kwargs: dict,
        metrics: list,
):

    evaluation_pipeline = Pipeline(
        pipeline_name="evaluation-pipeline",
        pipeline_description="Pipeline to evaluate \
        a RAG solution",
        base_path=pipeline_dir,  # The demo pipelines uses a local \
        # directory to store the data.
    )

    load_from_csv = ComponentOp(
        component_dir="components/load_from_csv",
        arguments={
            # Add arguments
            "dataset_uri": csv_dataset_uri,
            "column_separator": csv_column_separator,
            "column_name_mapping": {question_column_name: "text"},
        },
    )

    embed_text_op = ComponentOp.from_registry(
        name="embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
        },
    )

    retrieve_chunks = ComponentOp(
        component_dir="components/retrieve_from_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class_name,
            "top_k": top_k,
        },
    )

    retriever_eval = ComponentOp(
        component_dir="components/retriever_eval",
        arguments={
            "llm_name": llm_name,
            "llm_kwargs": llm_kwargs,
            "metrics": metrics,
        },
    )

    aggregate_results = ComponentOp(component_dir="components/aggregate_eval_results")

    # Construct your pipeline
    evaluation_pipeline.add_op(load_from_csv)
    evaluation_pipeline.add_op(embed_text_op, dependencies=load_from_csv)
    evaluation_pipeline.add_op(retrieve_chunks, dependencies=embed_text_op)
    evaluation_pipeline.add_op(retriever_eval, dependencies=retrieve_chunks)
    evaluation_pipeline.add_op(aggregate_results, dependencies=retriever_eval)

    return evaluation_pipeline

if __name__ == "__main__":
    pipeline = create_pipeline(
        pipeline_dir="./data-dir",
        embed_model_provider="huggingface",
        embed_model="all-MiniLM-L6-v2",
        weaviate_url="http://host.docker.internal:8080",
        weaviate_class_name="Pipeline_1",
        csv_dataset_uri="/data/wikitext_1000_q.csv", #make sure it is the same as mounted file
        csv_column_separator=";",
        question_column_name="question",
        top_k=3,
        llm_name="OpenAI",
        llm_kwargs={"openai_api_key": ""},
        metrics=["context_precision", "context_relevancy"]
    )
