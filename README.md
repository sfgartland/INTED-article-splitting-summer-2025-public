# Computational Analysis of Physics Education Research Articles

This project uses natural language processing (NLP) and machine learning techniques to analyse a corpus of physics education research articles. The main goal is to classify different sections of these articles (such as "Methods" and "Theoretical Framework") into granular categories and to analyse their content using text embeddings and dimensionality reduction.

**The full report from the project is available here:** [Article Splitting and Section Classification](./Article_Splitting_and_Section_Classification--INTED-UiO--Summer_2025.pdf)

---

## Computational Essays

This repository contains several computational essays in the form of Jupyter notebooks that walk through the different stages of the project.

#### [`raw_data_process.ipynb`](./raw_data_process.ipynb)

This notebook covers the initial data processing pipeline:

* **Data Loading and Filtering**: Loads the raw data and filters for relevant articles.
* **XML Parsing**: Parses the XML content of the articles to extract sections, including titles and content.
* **Data Structuring**: "Explodes" the data, creating a new dataframe where each row corresponds to a single section of an article for easier analysis.
* **Embedding Generation**: Generates text embeddings for the titles and content of each section using the `voyage-3-large` model from VoyageAI. This notebook creates two types of embeddings: one for the entire section and another for smaller sentence chunks within each section.
* **Validation**: Includes validation steps to ensure the quality of the extracted text and the consistency of the generated embeddings.

#### [`Initial_plots.ipynb`](./Initial_plots.ipynb)

This notebook focuses on the exploratory data analysis of the generated embeddings:

* **Dimensionality Reduction**: Applies t-SNE, PCA, and UMAP to reduce the dimensionality of the section embeddings for visualization.
* **Interactive Visualization**: Uses Plotly to create interactive 2D scatter plots of the reduced embeddings, allowing for the exploration of clusters and relationships between different sections.

### Section type classification

#### [`gpt-classifier.ipynb`](./section_type_classification/gpt-classifier.ipynb)

This notebook contains the legacy code for classifying article sections using GPT. While it has been largely superseded by the more abstract `LLM_classifier` framework, it is included as it was the original method used to generate the section classifications.

#### [`heuristics.ipynb`](./section_type_classification/heuristics.ipynb)

This notebook implements a probabilistic classification approach based on a set of heuristics. It combines evidence from a section`s title, content length, position within the article, and text embeddings to assign a classification, providing a baseline for comparison with the more advanced LLM-based methods.

#### [`bert_section_classifier.ipynb`](./bert_section_classifier.ipynb)

This notebook fine-tunes a SciBERT LLM for section classification under a Hugging Face framework, and evaluates resulting classification performance.

#### [`nonMLP_classifier.ipynb`](./nonMLP_classifier.ipynb)

This notebook implements a novel light-weight classification approach, leveraging the geoemetric properties of textual embeddings. See the full article for further details.

#### [`MLP_classifier.ipynb`](./MLP_classifier.ipynb)

This notebook further augments the approach outlined above with Multi-Layer Perceptrons, significantly enhancing performance at little additional computational cost. 

### Theory and method identification and classification

#### [`2D_projection_space`](./2D_projection_space.ipynb)

This notebook further explores the geometric properties of textual embeddings, analysing a variety of different properties in a variety of different ways.

#### [`Methods_Classification.ipynb`](./theory_and_methods_identification/Methods_Classification.ipynb)

This notebook details the process of classifying the "Methods" sections of the articles into more specific categories:

* **Category Discovery**: Uses a Large Language Model (LLM) to discover an initial set of categories for the research methods discussed in the articles.
* **Category Refinement**: The discovered categories are then reviewed and cleaned to create a more coherent and meaningful set of labels.
* **Batch Classification**: Classifies all "Methods" sections into the refined categories using a custom `MethodsSectionClassifier`.
* **Cost Estimation**: Includes a step to estimate the cost of using the OpenAI API for the classification task.

#### [`Theory_Classification.ipynb`](./theory_and_methods_identification/Theory_Classification.ipynb)

Similar to the [`Methods_Classification.ipynb`](./Methods_Classification.ipynb), this notebook focuses on classifying the "Theoretical Framework" sections of the articles:

* **Category Discovery and Refinement**: It also uses an LLM to discover and refine categories of theoretical frameworks.
* **Batch Classification**: A `FrameworkClassifier` is used to classify the "Theoretical Framework" sections into the established categories.
* **Cost Estimation**: This notebook also includes an API cost estimation for the classification.


---

## Modules and Scripts

The project also includes helper scripts and a reusable classification module.

### `LLM_classifier` Module

This repository contains the `LLM_classifier`, a Python module that provides an abstract and extensible framework for classifying text data using Large Language Models (LLMs). It is designed around a staged pipeline that includes category discovery, batch classification, category review, and re-classification to ensure high-quality, structured outputs. For more detailed information on its architecture and how to create custom classifiers, please see the module's [README](./LLM_classifier/README.md).

---

## How to Run

To run these computational essays, you will need to have `conda` installed. You can set up the environment with the required packages.

**Set API Keys:**
    You will need API keys for VoyageAI and OpenAI. Create a file named `.env.secret` in the root of the project directory and add your keys in the following format:
    ```
    VOYAGE_API_KEY="your_voyage_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    ```
