"""Fondant pipeline to index a RAG system."""
import pyarrow as pa
from fondant.pipeline import Pipeline, Resources


def create_pipeline(  # noqa: PLR0913
    pipeline_dir: str = "./data-dir",
    embed_model_provider: str = "huggingface",
    embed_model: str = "woody72/multilingual-e5-base",
    weaviate_url: str = "http://34.22.248.219:8080",
    weaviate_class_name: str = "BelgianLawDS",
    overwrite: bool = True,
    # indexing args
    # hf_dataset_name: str = "wikitext@~parquet",
    # data_column_name: str = "text",
    n_rows_to_load: int = None,
    csv_dataset_uri: str = "/data/articles.csv",
    csv_column_separator: str = None,
    data_column_name: str = "article",
    chunk_size: int = 512,
    chunk_overlap: int = 32,
):
    resources = Resources(accelerator_name="GPU", accelerator_number=1)

    indexing_pipeline = Pipeline(
        name="indexing-pipeline",
        description="Pipeline to prepare and process data for building a RAG solution",
        base_path=pipeline_dir,  # The demo pipelines uses a local directory to store the data.
    )

    # text = indexing_pipeline.read(
    #     "load_from_hf_hub",
    #     arguments={
    #         # Add arguments
    #         "dataset_name": hf_dataset_name,
    #         "column_name_mapping": {data_column_name: "text"},
    #         "n_rows_to_load": n_rows_to_load,
    #     },
    #     produces={
    #         "text": pa.string(),
    #     },
    # )

    load_from_csv = indexing_pipeline.read(
        "components/load_from_csv",
        arguments={
            # Add arguments
            "dataset_uri": csv_dataset_uri,
            "column_separator": csv_column_separator,
            "column_name_mapping": {
                "reference": "source",
                data_column_name: "text"
                },
            "n_rows_to_load": n_rows_to_load
        },
        resources=resources
    )

    chunks = load_from_csv.apply(
        "chunk_text",
        arguments={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        resources=resources
    )

    embeddings = chunks.apply(
        "embed_text",
        arguments={
            "model_provider": embed_model_provider,
            "model": embed_model,
        },
        resources=resources
    )

    embeddings.write(
        "index_weaviate",
        arguments={
            "weaviate_url": weaviate_url,
            "class_name": weaviate_class_name,
            "overwrite": overwrite,
        },
        # resources=resources
    )

    return indexing_pipeline
