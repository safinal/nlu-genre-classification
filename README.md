# Genre Classification Project Report (NLU, Fall 2024)


## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Tools and Datasets](#tools-and-datasets)
4. [Methodology](#methodology)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Abstract
This report outlines a genre classification project using natural language understanding (NLU) methods. A dataset of movie descriptions was analyzed and classified into 27 genres using various embeddings (TF-IDF, GloVe, Word2Vec, FastText, BERT, and RoBERTa) with a neural network. Results indicate that contextual embeddings like BERT outperform static ones, achieving the highest test accuracy of 76.80%. 

---

## Introduction
Genre classification of movies is an essential task in content-based recommendation systems and search engines. This study leverages NLU techniques to classify movies based on textual descriptions. The goal is to evaluate the performance of different embedding techniques and identify the most effective for this task.

---

## Tools and Datasets

### Tools
- **Programming Language**: Python
- **Libraries**: 
  - Preprocessing: `pandas`, `nltk`, `scikit-learn`
  - Embeddings: `gensim`, `transformers`, `sentence_transformers`
  - Machine Learning: `torch`

### Dataset
The dataset comprises movie descriptions with corresponding genres. 
- **Size**: 54,214 training samples and 27 genre labels.
- **Class Distribution**:
  - Largest class: Drama (13,613 samples, 25.1%)
  - Smallest class: War (132 samples, 0.24%)
- **Text Statistics**:
  - Mean description length: ~599 characters
  - Minimum length: 41 characters
  - Maximum length: 10,503 characters

---

## Methodology

### Preprocessing
1. **Text Normalization**: Lowercasing, removing special characters.
2. **Tokenization and Lemmatization**: Using `nltk`.
3. **Stopword Removal**: Using English stopword list.

### Embeddings
1. **TF-IDF**: Top 1000 features selected.
2. **Static Embeddings**:
   - GloVe (300d, Wikipedia and Gigaword corpus)
   - Word2Vec (300d, Google News)
   - FastText (300d, Wikipedia)
3. **Contextual Embeddings**:
   - BERT (`bert-base-uncased`)
   - RoBERTa (`roberta-base`)

### Neural Network Architecture
- **Model**: Linear layer with cross-entropy loss.
- **Training**: 
  - Optimizer: Adam
  - Epochs: 20
  - Batch Size: 128
  - Learning Rate: 0.001

---

## Results and Analysis

### Accuracy Comparison
| Embedding      | Train Accuracy (%) | Validation Accuracy (%) | Test Accuracy (%) |
|----------------|--------------------|--------------------------|--------------------|
| TF-IDF         | 73.26             | 69.52                   | 68.97             |
| GloVe          | 73.11             | 71.67                   | 71.96             |
| Word2Vec       | 70.96             | 70.33                   | 70.18             |
| FastText       | 63.83             | 63.58                   | 63.52             |
| BERT           | 79.59             | 77.06                   | 76.80             |
| RoBERTa        | 78.11             | 76.63                   | 76.30             |

### Analysis
1. **Contextual vs. Static Embeddings**: BERT achieved the best test accuracy (76.80%), demonstrating the superiority of contextual embeddings over static methods like GloVe and Word2Vec.
2. **TF-IDF Performance**: While simple, TF-IDF provided competitive results (68.97%) given its computational efficiency.
3. **FastText Limitations**: Underperformed due to its reliance on subword information, which may not capture higher-order semantics effectively.

---

## Conclusion
The study highlights that contextual embeddings like BERT significantly enhance genre classification accuracy compared to static embeddings and TF-IDF. Future work could explore fine-tuning BERT and leveraging multimodal inputs (e.g., movie posters) to improve classification further.

---

## References
1. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation.
2. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space.
3. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
4. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
