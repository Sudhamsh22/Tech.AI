# Models and Algorithms in Tech.AI

This document provides a detailed overview of the models and algorithms used in the **Tech.AI** laptop support assistant. The system is composed of two primary machine learning components: a PyTorch-based intent classifier and a custom lexical retriever.

---

## 1. Issue Classification Model

### Architecture
The classifier is implemented as a **Bag-of-Words Feed-Forward Neural Network** using PyTorch (`BagOfWordsClassifier`). 
- **Input Representation:** Text queries are tokenized and encoded into a Bag-of-Words (BoW) vector representation by a custom `Vectorizer`. The vector simply counts the frequency of each learned vocabulary token in the input string.
- **Layers**: 
  1. Fully Connected Layer (`nn.Linear`): Maps the `vocab_size` features to 128 hidden dimensions.
  2. Activation (`nn.ReLU`): Applies a non-linear ReLU transformation.
  3. Regularization (`nn.Dropout`): Applies dropout with a probability of 0.2 to prevent overfitting.
  4. Output Layer (`nn.Linear`): Maps the 128 hidden dimensions to `num_classes` (the number of issue categories like `connectivity`, `hardware_failure`, etc.).
- **Output:** The raw logits are passed through a `softmax` function during inference to obtain confidence probabilities for each class category.

### Training Algorithm
- **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
- **Optimizer:** Adam Optimizer (`torch.optim.Adam`) with a default learning rate of `0.01`.
- **Training Epochs:** Configured to run for 80 epochs over the seed training set.

---

## 2. Document Retrieval System

### Algorithm: BM25 (Best Matching 25)
The retrieval module (`BM25Retriever`) securely indexes and fetches relevant internal troubleshooting manuals and documents. It implements the classic **BM25 lexical ranking algorithm**, a state-of-the-art TF-IDF variant.

- **Tokenization:** Documents are processed into simple unigrams.
- **Scoring Details:**
  - Calculates Document Frequency (DF) and Inverse Document Frequency (IDF) across the knowledge base.
  - Scores query-document pairs by combining Term Frequency (TF) and IDF.
  - Controls term saturation using hyperparameters `$k_1 = 1.5$` and length normalization `$b = 0.75$`.
- **Result Output:** The `retrieve` function returns a ranked list of `RetrievedChunk` instances, selecting the top-$k$ highest-scoring evidence chunks to be used by the assistant.

---

## 3. Orchestration & Response Generation

The `TechSupportAssistantModel` integrates both components to provide a unified conversational response:
1. Predicts the issue category and calculates confidence using the **Classifier**.
2. Retrieves the most relevant documentation steps using the **BM25 Retriever**.
3. Dynamically assembles a step-by-step resolution plan (`SupportResponse`), citing the retrieved documents as primary evidence context for the agent's diagnosis.
