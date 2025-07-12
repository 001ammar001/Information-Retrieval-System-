# Information Retrieval (IR) System

A comprehensive Information Retrieval system that provides multiple search methods including TF-IDF, BERT embeddings, hybrid approaches, and topic detection. The system supports query suggestions and text processing capabilities.

## 🏗️ System Architecture

### Overview
The system follows a service orinted architecture with clear separation between online (real-time) and offline (training) services, and between each of the search methods:

```
IR System/
├── app/
│   ├── main.py                 # Flask API server
│   ├── services/               # Core services
│   ├── online/                 # Real-time search services
│   ├── offline/                # Training services
│   ├── db/                     # Database services
│   ├── joblib_files/           # Trained models storage
│   └── processed_datasets/     # Processed data storage
├── Pipfile                     # Python dependencies
└── *.ipynb                     # Jupyter notebooks for analysis
```

### Core Components

#### 1. **API Layer** (`app/main.py`)
- **Flask REST API** with CORS support
- **Endpoints:**
  - `/get_user_query/main_feature` - Main search functionality
  - `/get_user_query/secondary_features` - Secondary search features
  - `/get_search_suggestion` - Query suggestions
  - `/process_text` - Text preprocessing

#### 2. **Search Services** (`app/online/`)
- **TF-IDF Service** (`tfidf_service.py`) - Traditional vector space model
- **BERT Service** (`bert_service.py`) - Neural embeddings using BERT
- **Hybrid Service** (`hybrid_service.py`) - Combines TF-IDF and BERT
- **Topic Detection Service** (`topic_detection_service.py`) - Combines TF-IDF, BERT with topic detecttion
- **Query Suggestion Service** (`query_suggestion_service.py`) - query suggestions using pre-trained model 
- **Inverted Index Service** (`inverted_index_service.py`) - Inverted index used in TF-IDF for fast documents retrival

#### 3. **Core Services** (`app/services/`)
- **Search Service** (`search_service.py`) - Main search orchestrator
- **Text Processing Service** (`text_processing_services.py`) - Text preprocessing pipeline
- **TF-IDF Vectorizer** (`tf_idf_vectorizer.py`) - Custom TF-IDF implementation

#### 4. **Training Services** (`app/offline/`)
- **BERT Trainer** (`bert_trainer.py`) - BERT model training
- **TF-IDF Trainer** (`tfidf_trainer_service.py`) - TF-IDF model training
- **Topic Detection Trainer** (`topic_detection_trainer_service.py`) - LDA model training
- **Inverted Index Trainer** (`inverted_index_trainer.py`) - Index building
- **Query Suggestion Trainer** (`glove_wiki_query_suggestion_trainer.py`) - Query Suggestion model preparing

#### 5. **Database Layer** (`app/db/`)
- **PostgreSQL** database for document storage
- **Database Service** (`db_service.py`) - Database operations
- **Seeder Service** (`seeder_service.py`) - Data seeding utilities

## 🔄 Service Relationships

### Search Flow
1. **Query Processing**: Text preprocessing (spelling correction, lemmatization, stop word removal)
2. **Method Selection**: Choose search method (TF-IDF, BERT, Hybrid, Topic Detection)
3. **Document Retrieval**: Use inverted index for fast candidate selection
4. **Re-ranking**: Apply selected method for similarity scoring
5. **Result Formatting**: Return ranked documents with scores

### Training Pipeline
1. **Database Seeding**: Store documents in PostgreSQL
2. **Data Processing**: Convert datasets to JSON format (to use in index, bert training)
3. **Text Preprocessing**: Apply text processing pipeline
4. **Model Training**: Train respective models (TF-IDF, BERT, LDA, Word2Vec)
5. **Model Storage**: Save models to `joblib_files/` directory

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12
- PostgreSQL database
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IR-Project
```

### 2. Install Dependencies
```bash
# Install pipenv if not already installed
pip install pipenv

# Install project dependencies
pipenv install
```

### 3. Database Setup
```bash
# Set up PostgreSQL environment variables In database service
export DB_HOST=localhost
export DB_NAME=ir
export DB_USER=postgres
export DB_PASSWORD=1234awsd
export DB_PORT=5432

# Or create a .env file with these variables
```

### 4. Initialize Database
```bash
# Activate virtual environment
pipenv shell

# Run database seeder
python -m app.db.seeder_service
```

### 5. Data Processing & Model Training

#### Process Datasets
```bash
# Process datasets to JSON format
python -m app.offline.process_and_save_dataset_to_json
```

#### Train Models
```bash
# Train TF-IDF models
python -m app.offline.tfidf_trainer_service

# Train BERT models
python -m app.offline.bert_trainer

# Train topic detection models
python -m app.offline.topic_detection_trainer_service

# Train inverted index
python -m app.offline.inverted_index_trainer

# Train query suggestion model
python -m app.offline.glove_wiki_query_suggestion_trainer
```

### 6. Run the Application
```bash
# Start the Flask server
python app/main.py
```

The API will be available at `http://localhost:5000`

## 📊 Supported Datasets

- **Antique**: Question-answering dataset
- **Quora**: Question similarity dataset

## 🔍 API Endpoints

### Main Search
```
GET /get_user_query/main_feature?data_set_name={dataset}&query={query}&top_k={k}&method={method}
```

**Parameters:**
- `data_set_name`: "antique" or "quora"
- `query`: Search query
- `top_k`: Number of results (default: 5)
- `method`: "tfidf", "bert", "hybrid", or "topicdetection"

### Query Suggestions
```
GET /get_search_suggestion?data_set_name={dataset}&query={query}
```

### Text Processing
```
GET /process_text?query={query}
```

## 🎯 Search Methods

### 1. **TF-IDF**
- Traditional vector space model
- Fast and effective for keyword-based queries
- Uses cosine similarity for ranking

### 2. **BERT**
- Neural embeddings using BERT model
- Better semantic understanding
- Slower but more accurate for complex queries

### 3. **Hybrid**
- Combines TF-IDF and BERT (60% TF-IDF + 40% BERT)
- Balances speed and accuracy
- Uses TF-IDF for initial retrieval, BERT for re-ranking

### 4. **Topic Detection**
- Uses LDA (Latent Dirichlet Allocation) for topic modeling
- Combines TF-IDF, BERT, and topic scores
- Effective for thematic queries

## 📈 Performance Metrics

The system has been evaluated on both datasets:

## 🛠️ Development

### Project Structure
```
app/
├── online/          # Real-time services
├── offline/         # Training services
├── services/        # Core services
├── db/             # Database layer
├── joblib_files/   # Model storage
└── processed_datasets/  # Processed data
```

### Adding New Datasets
1. Add dataset to `processed_datasets/`
2. Update training scripts to include new dataset
3. Run training pipeline
4. Update API validation

### Adding New Search Methods
1. Create new service in `app/online/`
2. Implement `find_similar_documents()` method
3. Add method to `search_service.py`
4. Update API validation

## 🔧 Configuration

### Environment Variables
- `DB_HOST`: PostgreSQL host (default: localhost)
- `DB_NAME`: Database name (default: ir)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password (default: 1234awsd)
- `DB_PORT`: Database port (default: 5432)

### Model Storage
Trained models are stored in `app/joblib_files/{dataset_name}/`:
- `tf_idf_vectors.joblib` - TF-IDF models
- `bert_vectors.joblib` - BERT embeddings
- `topic_detection_vectors.joblib` - LDA models
- `inverted_index.joblib` - Inverted indices

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on the repository. 