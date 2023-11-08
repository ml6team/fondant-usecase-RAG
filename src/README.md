## 🍫 Building a RAG indexing pipeline with Fondant

This example demonstrates a Fondant data pipeline that ingests text
data into a vector database. The pipeline uses four reusable Fondant components.  
Additionally, we provide a Docker Compose setup for Weaviate, enabling local testing and
development.

### Pipeline overview

There are 4 components in total, these are:

- [Load from Huggingface Hub](https://github.com/ml6team/fondant/tree/main/components/load_from_hf_hub):
  The
  pipeline begins by loading text data from a Parquet file, which serves as the
  source for subsequent processing. For the minimal example we are using a dataset from Huggingface.
- [Text Chunking](https://github.com/ml6team/fondant/tree/main/components/chunk_text): Text data is
  chunked into manageable sections to prepare it for embedding. This
  step
  is crucial for performant RAG systems.
- [Text Embedding](https://github.com/ml6team/fondant/tree/main/components/embed_text): We are using
  a small HuggingFace model for the generation of text embeddings.
  The `embed_text` component easily allows the usage of different models as well.
- [Write to Weaviate](https://github.com/ml6team/fondant/tree/main/components/index_weaviate): The
  final step of the pipeline involves writing the embedded text data to
  a Weaviate database.

## Environment

Please check that the following prerequisites are:
- A python version between 3.8 and 3.10 is installed on your system
  ```shell
  python --version
  ```
- Docker compose is installed on your system and the docker daemon is running
  ```shell
  docker compose version
  docker info
  ```
  
- Fondant is installed
  ```shell
  fondant
  ```

## Implementing the pipeline

The pipeline is implemented in [pipeline.py](pipeline.py). Please have a look at the file so you 
understand what is happening.

For more details on the pipeline creation, you can have a look at the 
[pipeline.ipynb](pipeline.ipynb) notebook which describes the process step by step.

## Running the pipeline

This pipeline will load, chunk and embed text, as well as ingest the text embeddings to a vector
database.

Fondant provides multiple runners to run our pipeline:
- A Docker runner for local execution
- A Vertex AI runner for managed execution on Google Cloud
- A Kubeflow Pipelines runner for execution anywhere

Here we will use the local runner, which utilizes Docker compose under the hood.

The runner will first build the custom component and download the reusable components from the 
component hub. Afterwards, you will see the components execute one by one.

```shell
fondant run local pipeline.py
```

## Exploring the dataset

You can explore the dataset using the fondant explorer, this enables you to visualize your output 
dataset at each component step. Use the side panel on the left to browse through the steps and subsets.

```shell
fondant explore -b data-dir
```

## Scaling up

If you're happy with your dataset, it's time to scale up. Check 
[our documentation](https://fondant.ai/en/latest/pipeline/#compiling-and-running-a-pipeline) for 
more information about the available runners.
