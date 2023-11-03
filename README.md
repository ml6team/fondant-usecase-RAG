## 🍫 Building a RAG indexing pipeline with Fondant

This repository demonstrates a Fondant data pipeline that ingests text
data into a vector database. The pipeline uses four reusable Fondant components.  
Additionally, we provide a Docker Compose setup for Weaviate, enabling local testing and
development.

### Pipeline overview

The primary goal of this sample is to showcase how you can use a Fondant pipeline and reusable
components to load, chunk and embed text, as well as ingest the text embeddings to a vector
database.
Pipeline Steps:

- [Data Loading](https://github.com/ml6team/fondant/tree/main/components/load_from_parquet): The
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

### Usage

**Prerequisite:**

Ensure that you have Python installed on your system. This data pipeline requires Python version 3.8
to 3.10.
Docker is necessary to run the pipeline and the Weaviate database locally. Ensure that Docker is
installed and configured on your system.

Follow these steps to get started and running the Fondant pipeline on your local machine.

1. **Setup your environment:** Clone this repository to your local machine using the following
   command:

```shell
git clone https://github.com/ml6team/fondant-usecase-RAG
```

or use SSH instead:

```shell
git clone git@github.com:ml6team/fondant-usecase-RAG.git
```

Afterwards, you can go into the `/src` folder and install all needed requirements:

```shell
cd src
```

```shell
pip install -r requirements.txt
```

Please confirm that Fondant has been installed correctly on your system by executing the following
command:

```shell
fondant --help
```

2. **Start the vector database**: Navigate to the `weaviate` directory and start the Weaviate
   instance using Docker Compose:

```shell
cd weaviate 
docker compose up
```

Ensure that the database instance is running by validating access with the following command:

```shell
curl http://localhost:8080/v1/meta
```

3. **Run the pipeline:** Please navigate to the root directory of this repository and perform the
   following:

```shell
fondant run local pipeline.py
```

The pipeline will be compiled into a `docker-compose.yaml` file and subsequently executed.

Fondant provides various runners to execute the pipeline in different environments. If you intend to
run the pipeline in a production environment, you can utilize, for example,
the [VertexAI runner](https://fondant.ai/en/latest/pipeline/#vertex-runner).
