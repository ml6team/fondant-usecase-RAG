# Retrieval Augmented Generation (RAG) Tuning

<p align="center">
    <a href="https://github.com/ml6team/fondant">
        <img src="https://raw.githubusercontent.com/ml6team/fondant/main/docs/art/fondant_banner.svg" height="150px"/>
    </a>
</p>
<p align="center">
</p>
<p align="center">
</p>
<p align="center"><b>
Check out our <a href="https://medium.com/fondant-blog/lets-tune-rag-pipelines-with-fondant-902f7215e540">blogpost</a> about how to to fine-tune your RAG pipeline using Fondant!</b></p>

## Introduction

This repository contains data pipelines and ready-to-use notebooks for tuning RAG systems both manually and automatically using parameter search.
To achieve this, it leverages [Fondant](https://github.com/ml6team/fondant), a free and open source framework for production-ready, easy and shareable data processing. 
Check out the Fondant [website](https://fondant.ai/) if you want to learn more and join our [Discord](https://discord.gg/HnTdWhydGp) if you want to stay up to date.

## Available notebooks

### A simple RAG indexing pipeline

A [**notebook**](./src/indexing.ipynb) with a simple Fondant pipeline to index your data into a 
RAG system.

### Iterative tuning of a RAG indexing pipeline

A [**notebook**](./src/evaluation.ipynb) which iteratively runs a Fondant
pipeline to evaluate a RAG system using [RAGAS](https://github.com/explodinggradients/ragas/tree/main/src/ragas).

## Getting started

> ⚠️ **Prerequisites:**
>
> - A Python version between 3.8 and 3.10 installed on your system.
> - Docker and docker compose installed and configured on your system. More info [here](https://fondant.ai/en/latest/guides/installation/#docker-installation).
> - A GPU is recommended to run the model-based components of the pipeline.

### Cloning the repository

Clone this repository to your local machine using one of the following commands:

**HTTPS**
```shell
git clone https://github.com/ml6team/fondant-usecase-rag.git
```

**SSH**
```shell
git clone git@github.com:ml6team/fondant-usecase-rag.git
```

### Installing the requirements

```shell
pip install -r requirements.txt
```

Confirm that Fondant has been installed correctly on your system by executing the following command:

```shell
fondant --help
```

### Running the pipeline

There are two options to run the pipeline:

- [**Via python files and the Fondant CLI:**](https://fondant.ai/en/latest/pipeline/#running-a-pipeline) how you should run Fondant in production
- [**Via a Jupyter notebook**](./src/indexing.ipynb): ideal to learn about Fondant
