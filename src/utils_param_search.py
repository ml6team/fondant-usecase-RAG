import itertools
import logging
from datetime import datetime

import pipeline_eval
import pipeline_index
import weaviate
from fondant.pipeline.runner import DockerRunner
from utils import read_latest_data


def run_parameters_search(  # noqa: PLR0913
    extra_volumes,
    fixed_args,
    fixed_index_args,
    fixed_eval_args,
    chunk_sizes,
    chunk_overlaps,
    embed_models,
    top_ks,
):
    # Define pipeline runner
    runner = DockerRunner()

    # Results dictionary to store results for each iteration
    results_dict = {}

    # Perform grid search
    indexes = []
    for i, (chunk_size, chunk_overlap, embed_model) in enumerate(
        itertools.product(chunk_sizes, chunk_overlaps, embed_models),
        start=1,
    ):
        index_config_class_name = f"IndexConfig{i}"
        index_pipeline_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            f"Running indexing for {index_config_class_name} \
            with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\
            , embed_model={embed_model}",
        )

        # Store Indexing configuration
        index_dict = {}
        index_dict["index_name"] = index_config_class_name
        index_dict["indexing_datetime"] = index_pipeline_datetime
        index_dict["chunk_size"] = chunk_size
        index_dict["chunk_overlap"] = chunk_overlap
        index_dict["embed_model"] = embed_model
        indexes.append(index_dict)

        # Create and Run the indexing pipeline
        indexing_pipeline = pipeline_index.create_pipeline(
            **fixed_args,
            **fixed_index_args,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_provider=embed_model[0],
            embed_model=embed_model[1],
            weaviate_class=index_config_class_name,
        )
        run_indexing_pipeline(
            runner=runner,
            index_pipeline=indexing_pipeline,
            weaviate_url=fixed_args["weaviate_url"],
            weaviate_class=index_config_class_name,
        )

    parameters_search_results = []
    for i, (index_dict, top_k) in enumerate(
        itertools.product(indexes, top_ks),
        start=1,
    ):
        rag_config_name = f"RAGConfig{i}"
        eval_pipeline_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            f"Running evaluation for {rag_config_name} \
            with {index_dict['index_name']} and {top_k} retrieved chunks",
        )

        # Store RAG pipeline configuration
        results_dict = {}
        results_dict["rag_config_name"] = rag_config_name
        results_dict["evaluation_datetime"] = eval_pipeline_datetime
        results_dict.update(index_dict)
        results_dict["top_k"] = top_k

        # Create and Run the evaluation pipeline
        evaluation_pipeline = pipeline_eval.create_pipeline(
            **fixed_args,
            **fixed_eval_args,
            embed_model_provider=index_dict["embed_model"][0],
            embed_model=index_dict["embed_model"][1],
            weaviate_class=index_dict["index_name"],
            retrieval_top_k=top_k,
        )
        run_evaluation_pipeline(
            runner=runner,
            eval_pipeline=evaluation_pipeline,
            extra_volumes=extra_volumes,
        )

        # Save the evaluation results in the dictionary
        results_dict[f"agg_results_{rag_config_name}"] = read_latest_data(
            base_path=fixed_args["base_path"],
            pipeline_name="evaluation-pipeline",
        )
        # Add fixed arguments
        results_dict.update(fixed_args)
        results_dict.update(fixed_index_args)
        results_dict.update(fixed_eval_args)

        parameters_search_results.append(results_dict)

    return parameters_search_results


# index pipeline runner
def run_indexing_pipeline(runner, index_pipeline, weaviate_url, weaviate_class):
    runner.run(index_pipeline)
    docker_weaviate_client = weaviate.Client(weaviate_url)
    return docker_weaviate_client.schema.get(weaviate_class)


# eval pipeline runner
def run_evaluation_pipeline(runner, eval_pipeline, extra_volumes):
    runner.run(input=eval_pipeline, extra_volumes=extra_volumes)
