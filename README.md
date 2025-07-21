# News Classification and NLP Analysis: A Comprehensive Text Processing Pipeline

## Overview
This project implements a complete Natural Language Processing pipeline for Spanish news articles, covering web scraping, text preprocessing, classification, similarity analysis, and automatic summarization. The system processes news from four different categories and applies various NLP techniques to analyze and classify text content.

## Project Objectives
- **Web Scraping**: Extract news articles from multiple Spanish websites
- **Text Classification**: Train models to automatically categorize news articles
- **Text Similarity Analysis**: Compare document similarity using different embedding approaches
- **Automatic Summarization**: Generate extractive summaries for news content
- **Interactive Systems**: Create user-friendly interfaces including a Telegram bot

## Key Features

### 1. Web Scraping Module
- Automated data collection from 4 Spanish websites
- Ethical scraping with robots.txt compliance
- Rate limiting to avoid server overload
- 40 articles collected across 4 categories:
  - **Economy & Finance** (10 articles)
  - **Entertainment** (10 articles) 
  - **Fitness & Sports** (10 articles)
  - **Medicine & Health** (10 articles)

### 2. Text Preprocessing Pipeline
- **Normalization**: Lowercase conversion, accent removal, punctuation cleaning
- **Stopword Removal**: Spanish stopwords elimination
- **Abbreviation Expansion**: Standardization of common abbreviations
- **Spell Checking**: Orthographic correction using SpellChecker
- **Lemmatization**: Word reduction to base forms using spaCy
- **Special Character Handling**: Emoji and symbol processing

### 3. Classification Models
Two different vectorization approaches implemented:

#### TF-IDF Vectorization + Logistic Regression
- Traditional term frequency-inverse document frequency approach
- Logistic Regression classifier
- **Performance**: 62.5% accuracy
- Best performance on Economy & Finance category

#### BERT Embeddings + Logistic Regression  
- Spanish BERT model: `dccuchile/bert-base-spanish-wwm-cased`
- Semantic embedding approach
- **Performance**: 62.5% accuracy
- Enhanced semantic understanding

### 4. Similarity Analysis
Two embedding models for document similarity:

#### Doc2Vec
- Document-level embeddings for title comparison
- Similarity scores ranging from 0.31 to 0.80
- Effective for capturing document structure

#### Universal Sentence Encoder
- Multilingual sentence-level embeddings
- Advanced semantic similarity detection
- Interactive similarity matrix visualization

### 5. Automatic Summarization
- **Extractive Approach**: PageRank-based sentence selection
- **Similarity Matrix Analysis**: Cosine similarity between sentences
- **Configurable Length**: 3-sentence summaries per article
- **Quality Assurance**: Preserves original content integrity

### 6. Interactive Applications
- **Command-line Interface**: Category-based news summary retrieval
- **Telegram Bot**: Real-time news summarization service
- **User-friendly Workflow**: Guided category selection

## Technical Implementation

### Development Environment
- **Platform**: Google Colab
- **Format**: Single Jupyter Notebook (.ipynb)
- **Language**: Python 3.x
- **Environment**: Cloud-based with GPU support available

### Libraries & Dependencies
All dependencies were installed directly in the Colab environment using:

```python
# Core NLP Libraries
!pip install spacy nltk transformers torch
!python3 -m spacy download es_core_news_sm
!python3 -m spacy download es_core_news_md

# Web Scraping & Data Processing
!pip install selenium beautifulsoup4 pandas unidecode

# Machine Learning & Analysis
!pip install scikit-learn autocorrect pyspellchecker textblob

# Visualization & NLP Tools
!pip install wordcloud matplotlib plotly gensim bokeh
!pip install tensorflow-text tensorflow-hub networkx seaborn

# Optional: Telegram Bot
!pip install pyTelegramBotAPI
```

### Project Structure
```
TP1_Procesamiento_del_lenguaje_natural.ipynb
├── Exercise 1: Web Scraping
│   ├── Website permissions verification
│   ├── Data extraction from 4 Spanish websites
│   └── CSV dataset creation (40 articles)
├── Exercise 2: Text Classification
│   ├── Text preprocessing pipeline
│   ├── TF-IDF vectorization + Logistic Regression
│   ├── BERT embeddings + Logistic Regression
│   └── Model evaluation and comparison
├── Exercise 3: Content Analysis
│   ├── Full article text processing
│   ├── Word frequency analysis
│   └── Word cloud generation per category
├── Exercise 4: Similarity Analysis
│   ├── Doc2Vec implementation
│   ├── Universal Sentence Encoder
│   └── Similarity matrix visualization
└── Exercise 5: Text Summarization
    ├── Extractive summarization using PageRank
    ├── Interactive summary generator
    └── Optional: Telegram bot implementation
```

### Colab-Specific Features
- **Runtime Management**: Optimized for Colab's session limitations
- **GPU Acceleration**: BERT model processing with CUDA support
- **File Handling**: Direct CSV export and data persistence
- **Visualization**: Inline plots and interactive widgets
- **Memory Management**: Efficient handling of large language models

## Key Results & Insights

### Classification Performance
- **Overall Accuracy**: 62.5% for both TF-IDF and BERT approaches
- **Category-specific Performance**:
  - Economy & Finance: Excellent precision with TF-IDF
  - Medicine & Health: Strong recall with BERT embeddings
  - Entertainment & Sports: Moderate performance, semantic overlap detected

### Text Analysis Findings
- **Word Frequency Analysis**: Clear category-specific vocabulary patterns
- **Semantic Relationships**: Medicine & Health + Fitness & Sports show high semantic similarity
- **Preprocessing Impact**: Significant improvement after stopword removal and normalization

### Similarity Analysis Results
- **Doc2Vec Similarities**: 0.31 - 0.80 range across fitness/sports articles
- **Universal Sentence Encoder**: Enhanced semantic understanding with different similarity patterns
- **Cross-model Validation**: Consistent high-similarity pairs across different embedding approaches

## Dataset Information
- **Total Articles**: 40 news articles
- **Languages**: Spanish
- **Sources**: 4 reputable Spanish websites
- **Categories**: Balanced distribution (10 articles per category)
- **Content**: Full article text + metadata (URL, title, category)
- **Output Format**: CSV file (`noticias.csv`) generated in Exercise 1
- **Data Storage**: Saved directly in Google Colab environment

## File Structure
The project consists of a single comprehensive Jupyter notebook:
- **Main File**: `TP1_Procesamiento_del_lenguaje_natural_Fernandez_Palermo_Salvaña.ipynb`
- **Generated Output**: `noticias.csv` (news dataset)
- **Visualizations**: Inline plots, word clouds, and similarity matrices
- **Models**: Trained in-memory (TF-IDF vectorizers, BERT embeddings)

## Ethical Considerations
- **robots.txt Compliance**: All websites checked for scraping permissions
- **Rate Limiting**: 10-second delays between requests
- **Personal Use**: Data collection limited to educational/research purposes
- **Content Integrity**: Extractive summarization preserves original meaning

## Future Improvements
- **Dataset Expansion**: Include more articles and categories
- **Advanced Models**: Experiment with transformer-based classifiers
- **Real-time Processing**: Implement streaming news analysis
- **Multilingual Support**: Extend to other languages
- **Performance Optimization**: GPU acceleration for BERT processing

## Academic Context
This project was developed as part of a Natural Language Processing course, demonstrating practical application of:
- Web scraping techniques and ethical considerations
- Comprehensive text preprocessing pipelines
- Comparative analysis of vectorization approaches
- Document similarity and clustering methods
- Extractive summarization algorithms
- Interactive application development

## Usage Examples

### Running the Complete Pipeline
The entire project is contained in a single Google Colab notebook. To run:

1. **Open in Google Colab**: Upload the `.ipynb` file to Google Colab
2. **Install Dependencies**: Run the installation cells at the beginning
3. **Execute Sequentially**: Run cells in order from Exercise 1 through 5
4. **Data Persistence**: Generated CSV and models are saved in the Colab environment

### Key Code Snippets

#### Web Scraping (Exercise 1)
```python
# Extract news articles from Spanish websites
from selenium import webdriver
from bs4 import BeautifulSoup

# Automated scraping with ethical considerations
driver = webdriver.Chrome(options=chrome_options)
# Rate limiting and robots.txt compliance implemented
```

#### Classification Pipeline (Exercise 2)
```python
# TF-IDF approach
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer()
X_train_vectorized = tfidf.fit_transform(X_train_list)
modelo_LR = LogisticRegression(max_iter=1000)
modelo_LR.fit(X_train_vectorized, y_train)
```

#### BERT Implementation (Exercise 2)
```python
# Spanish BERT embeddings
from transformers import BertTokenizer, BertModel
model_name = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
```

#### Similarity Analysis (Exercise 4)
```python
# Doc2Vec implementation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs=1000)
similares = model.docvecs.most_similar([model.dv[i]], topn=10)
```

#### Extractive Summarization (Exercise 5)
```python
# PageRank-based summarization
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph, max_iter=50000, tol=1.0e-1)
```

### Notebook Execution Notes
- **Runtime**: Approximately 2-3 hours for complete execution
- **Memory Requirements**: High RAM runtime recommended for BERT processing
- **GPU Usage**: Optional but recommended for faster BERT inference
- **Data Output**: `noticias.csv` file generated in Exercise 1

---

**Note**: This project demonstrates the complete NLP pipeline from data collection to deployment, showcasing both traditional and modern approaches to text analysis in Spanish language processing.
