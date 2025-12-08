import os
import re
import json
import requests
import pdfplumber
import numpy as np
import pandas as pd
import chromadb
import nltk
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

BOOK_URLS = {
    "An Introduction to Data Science": "https://mrcet.com/downloads/digital_notes/CSE/II%20Year/DS/Introduction%20to%20Datascience%20%5BR20DS501%5D.pdf",
    "Data Science for Beginners": "https://slims.ahmaddahlan.ac.id/index.php?p=fstream-pdf&fid=53&bid=3197",
    "Data Science": "https://mrce.in/ebooks/Data%20Science.pdf",
    "Introducing Microsoft Power BI": "https://kh.aquaenergyexpo.com/wp-content/uploads/2024/03/Introducing-Microsoft-Power-BI.pdf",
    "Power BI for Beginners": "https://www.data-action-lab.com/wp-content/uploads/2024/01/Power-BI-for-Beginners.pdf"
}

DOWNLOAD_DIR = "./books"
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000 
TOP_K_RESULTS = 5
PCA_COMPONENTS = 100  #Number of the pca component has been increased
MIN_DF = 3  
MAX_DF = 0.7  
N_GRAMS = (1, 3)  

class BookDownloader:
    def __init__(self, download_dir: str = DOWNLOAD_DIR):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_books(self) -> Dict[str, str]:
        book_paths = {}

        for book_name, url in tqdm(BOOK_URLS.items(), desc="Downloading books"):
            try:
                filename = f"{book_name.replace(' ', '_').replace(':', '').replace('[', '').replace(']', '').lower()}.pdf"
                filepath = os.path.join(self.download_dir, filename)

                if os.path.exists(filepath):
                    book_paths[book_name] = filepath
                    continue

                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                book_paths[book_name] = filepath
                print(f"Downloaded: {book_name}")

            except Exception as e:
                print(f"Failed to download {book_name}: {str(e)}")

        return book_paths

class PDFTextExtractor:

    def __init__(self):
        self.download_dir = DOWNLOAD_DIR

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        text_by_page = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:

                        cleaned_text = self._clean_extracted_text(text)
                        if cleaned_text:
                            text_by_page.append({
                                'text': cleaned_text,
                                'page': page.page_number,
                                'source': os.path.basename(pdf_path)
                            })
        except Exception as e:
            print(f"Error extracting text with pdfplumber from {pdf_path}: {str(e)}")

            text_by_page = self._extract_text_with_ocr(pdf_path)

        return text_by_page

    def _clean_extracted_text(self, text: str) -> str:
        """Clean text immediately after extraction"""

        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'Page \d+ of \d+', '', text)  # Page X of Y
        text = re.sub(r'Chapter \d+', '', text)  # Chapter X

        text = re.sub(r'\s+', ' ', text).strip()

        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.split()) > 3]
        text = ' '.join(filtered_lines)

        return text

    def _extract_text_with_ocr(self, pdf_path: str) -> List[Dict]:
        """Extract text using OCR for scanned PDFs"""
        text_by_page = []

        try:
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                cleaned_text = self._clean_extracted_text(text)
                if cleaned_text:
                    text_by_page.append({
                        'text': cleaned_text,
                        'page': i + 1,
                        'source': os.path.basename(pdf_path)
                    })
        except Exception as e:
            print(f"OCR extraction failed for {pdf_path}: {str(e)}")

        return text_by_page

    def extract_all_books(self, book_paths: Dict[str, str]) -> List[Dict]:
        """Extract text from all books"""
        all_text = []

        for book_name, path in tqdm(book_paths.items(), desc="Extracting text"):
            text_by_page = self.extract_text_from_pdf(path)
            all_text.extend(text_by_page)

        return all_text

class TextPreprocessor:
    """Enhanced text preprocessing with more aggressive cleaning"""

    def __init__(self, min_word_length=3, max_word_length=20):
        self.stemmer = nltk.stem.PorterStemmer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

        custom_stopwords = ['fig', 'figure', 'table', 'et', 'al', 'etc', 'ie', 'eg', 'chapter', 'section']
        self.stop_words.update(custom_stopwords)
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep some punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)

        return text

    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]

    def filter_by_length(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens
                if self.min_word_length <= len(token) <= self.max_word_length]

    def remove_rare_words(self, tokens: List[str], min_freq: int = 2) -> List[str]:
        word_freq = Counter(tokens)
        return [token for token in tokens if word_freq[token] >= min_freq]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, text: str, use_stemming=True, use_lemmatization=False) -> Dict:

        cleaned_text = self.clean_text(text)

        tokens = self.tokenize(cleaned_text)

        filtered_tokens = self.remove_stopwords(tokens)

        length_filtered_tokens = self.filter_by_length(filtered_tokens)

        rare_filtered_tokens = self.remove_rare_words(length_filtered_tokens)

        if use_stemming and use_lemmatization:
            stemmed_tokens = self.stem_tokens(rare_filtered_tokens)
            final_tokens = self.lemmatize_tokens(stemmed_tokens)
        elif use_stemming:
            final_tokens = self.stem_tokens(rare_filtered_tokens)
        elif use_lemmatization:
            final_tokens = self.lemmatize_tokens(rare_filtered_tokens)
        else:
            final_tokens = rare_filtered_tokens

        pos_tags = nltk.pos_tag(length_filtered_tokens)

        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'length_filtered_tokens': length_filtered_tokens,
            'rare_filtered_tokens': rare_filtered_tokens,
            'final_tokens': final_tokens,
            'pos_tags': pos_tags
        }

class DataAnalyzer:
    """Performs exploratory data analysis on text data"""

    def __init__(self):
        pass

    def analyze_text_data(self, texts: List[str]) -> Dict:

        # Basic statistics
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]

        # Vocabulary analysis
        all_words = ' '.join(texts).split()
        word_freq = Counter(all_words)
        vocab_size = len(word_freq)

        # Most common words
        most_common_words = word_freq.most_common(20)

        # Sentence analysis
        sentence_counts = [len(nltk.sent_tokenize(text)) for text in texts]

        # Document length distribution
        doc_lengths = [len(text) for text in texts]

        return {
            'word_count_stats': {
                'mean': np.mean(word_counts),
                'median': np.median(word_counts),
                'min': np.min(word_counts),
                'max': np.max(word_counts),
                'std': np.std(word_counts)
            },
            'char_count_stats': {
                'mean': np.mean(char_counts),
                'median': np.median(char_counts),
                'min': np.min(char_counts),
                'max': np.max(char_counts),
                'std': np.std(char_counts)
            },
            'sentence_count_stats': {
                'mean': np.mean(sentence_counts),
                'median': np.median(sentence_counts),
                'min': np.min(sentence_counts),
                'max': np.max(sentence_counts),
                'std': np.std(sentence_counts)
            },
            'doc_length_stats': {
                'mean': np.mean(doc_lengths),
                'median': np.median(doc_lengths),
                'min': np.min(doc_lengths),
                'max': np.max(doc_lengths),
                'std': np.std(doc_lengths)
            },
            'vocab_size': vocab_size,
            'most_common_words': most_common_words,
            'word_freq': word_freq
        }

    def plot_word_frequency(self, word_freq: Counter, top_n=20):

        most_common = word_freq.most_common(top_n)
        words = [word for word, _ in most_common]
        counts = [count for _, count in most_common]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts, y=words)
        plt.title(f'Top {top_n} Most Common Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.tight_layout()
        plt.show()

    def plot_text_length_distribution(self, texts: List[str]):

        word_counts = [len(text.split()) for text in texts]

        plt.figure(figsize=(12, 6))
        sns.histplot(word_counts, kde=True)
        plt.title('Text Length Distribution (Word Count)')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

class FeatureProcessor:

    def __init__(self, n_components: int = PCA_COMPONENTS,
                 vectorizer_type: str = 'tfidf',
                 dim_reduction: str = 'pca'):
        self.n_components = n_components
        self.vectorizer_type = vectorizer_type
        self.dim_reduction = dim_reduction

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=MIN_DF,
                max_df=MAX_DF,
                ngram_range=N_GRAMS,
                sublinear_tf=True,
                smooth_idf=True)

        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=2000,
                min_df=MIN_DF,
                max_df=MAX_DF,
                ngram_range=N_GRAMS
            )
        else:
            raise ValueError("Vectorizer type must be 'tfidf' or 'count'")

        if dim_reduction == 'pca':
            self.reducer = PCA(n_components=n_components, random_state=42)
            self.scaler = StandardScaler()
        elif dim_reduction == 'svd':
            self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Dimensionality reduction must be 'pca' or 'svd'")

        self.feature_selector = SelectKBest(f_classif, k=min(500, n_components))

    def vectorize_texts(self, texts: List[str]) -> np.ndarray:

        return self.vectorizer.fit_transform(texts).toarray()

    def apply_dimensionality_reduction(self, features: np.ndarray) -> np.ndarray:

        scaled_features = self.scaler.fit_transform(features)

        reduced_features = self.reducer.fit_transform(scaled_features)

        return reduced_features

    def apply_feature_selection(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:

        return self.feature_selector.fit_transform(features, labels)

    def process_features(self, texts: List[str], labels: Optional[np.ndarray] = None,
                        apply_dim_reduction: bool = True, apply_feature_selection: bool = True) -> np.ndarray:

        # Vectorize texts
        features = self.vectorize_texts(texts)

        if apply_feature_selection and labels is not None:
            features = self.apply_feature_selection(features, labels)

        if apply_dim_reduction:
            features = self.apply_dimensionality_reduction(features)

        return features

    def get_feature_names(self) -> List[str]:

        return self.vectorizer.get_feature_names_out()

    def get_explained_variance(self) -> np.ndarray:

        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        return np.array([])

class EmbeddingGenerator:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        return self.model.encode(texts)

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 100) -> List[str]:

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

class VectorDatabase:


    def __init__(self, db_path: str = VECTOR_DB_PATH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("books_collection")

    def add_documents(self, documents: List[str], embeddings: np.ndarray,
                     metadatas: List[Dict], ids: List[str]) -> None:

        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embedding: np.ndarray, n_results: int = TOP_K_RESULTS,
              where: Optional[Dict] = None) -> Dict:

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )

    def count(self) -> int:

        return self.collection.count()

    def get_all_documents(self) -> Dict:
        return self.collection.get()

    def delete_documents(self, ids: List[str]) -> None:

        self.collection.delete(ids=ids)

    def update_documents(self, ids: List[str], documents: List[str] = None,
                        embeddings: np.ndarray = None, metadatas: List[Dict] = None) -> None:
        update_dict = {}

        if documents is not None:

            update_dict['documents'] = documents

        if embeddings is not None:

            update_dict['embeddings'] = embeddings.tolist()

        if metadatas is not None:

            update_dict['metadatas'] = metadatas

        self.collection.update(ids=ids, **update_dict)

class MLModelTrainer:

    def __init__(self, n_components: int = PCA_COMPONENTS):
        self.models = {
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.feature_processor = FeatureProcessor(n_components)
        self.best_models = {}
        self.model_results = {}
        self.label_encoder = LabelEncoder()

    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple:
        """Prepare data for training with different feature processing for each model"""

        encoded_labels = self.label_encoder.fit_transform(labels)

        X_pca = self.feature_processor.process_features(texts, encoded_labels, apply_dim_reduction=True)

        X_tfidf = self.feature_processor.vectorize_texts(texts)

        X_train_pca, X_test_pca, y_train, y_test = train_test_split(
            X_pca, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )

        X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
            X_tfidf, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )

        return (X_train_pca, X_test_pca, X_train_tfidf, X_test_tfidf, y_train, y_test)

    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Perform hyperparameter tuning for a specific model"""
        model = self.models[model_name]

        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['linear', 'rbf', 'poly'],
                'class_weight': ['balanced', None]
            },
            'Naive Bayes': {
                'alpha': [0.01, 0.1, 0.5, 1.0, 1.5],
                'fit_prior': [True, False]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

    def train_and_evaluate(self, texts: List[str], labels: List[str],
                          tune_hyperparameters: bool = True) -> Dict:

        X_train_pca, X_test_pca, X_train_tfidf, X_test_tfidf, y_train, y_test = self.prepare_data(texts, labels)

        results = {}

        if tune_hyperparameters:
            svm_tuning = self.hyperparameter_tuning('SVM', X_train_pca, y_train)
            svm_model = svm_tuning['best_model']
            best_params = svm_tuning['best_params']
        else:
            svm_model = self.models['SVM']
            best_params = "Default"

        svm_model.fit(X_train_pca, y_train)
        y_pred_svm = svm_model.predict(X_test_pca)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(svm_model, X_train_pca, y_train, cv=cv, scoring='f1_weighted')

        results['SVM'] = {
            'accuracy': accuracy_score(y_test, y_pred_svm),
            'f1_score': classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)['weighted avg']['f1-score'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_svm),
            'best_params': best_params,
            'features_used': 'PCA'
        }

        self.best_models['SVM'] = svm_model

        if tune_hyperparameters:
            nb_tuning = self.hyperparameter_tuning('Naive Bayes', X_train_tfidf, y_train)
            nb_model = nb_tuning['best_model']
            best_params = nb_tuning['best_params']
        else:
            nb_model = self.models['Naive Bayes']
            best_params = "Default"

        nb_model.fit(X_train_tfidf, y_train)
        y_pred_nb = nb_model.predict(X_test_tfidf)

        cv_scores = cross_val_score(nb_model, X_train_tfidf, y_train, cv=cv, scoring='f1_weighted')

        results['Naive Bayes'] = {
            'accuracy': accuracy_score(y_test, y_pred_nb),
            'f1_score': classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)['weighted avg']['f1-score'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_nb),
            'best_params': best_params,
            'features_used': 'TF-IDF'
        }

        self.best_models['Naive Bayes'] = nb_model

        if tune_hyperparameters:
            rf_tuning = self.hyperparameter_tuning('Random Forest', X_train_pca, y_train)
            rf_model = rf_tuning['best_model']
            best_params = rf_tuning['best_params']
        else:
            rf_model = self.models['Random Forest']
            best_params = "Default"

        rf_model.fit(X_train_pca, y_train)
        y_pred_rf = rf_model.predict(X_test_pca)

        cv_scores = cross_val_score(rf_model, X_train_pca, y_train, cv=cv, scoring='f1_weighted')

        results['Random Forest'] = {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'f1_score': classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)['weighted avg']['f1-score'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
            'best_params': best_params,
            'features_used': 'PCA'
        }

        self.best_models['Random Forest'] = rf_model

        if tune_hyperparameters:
            lr_tuning = self.hyperparameter_tuning('Logistic Regression', X_train_pca, y_train)
            lr_model = lr_tuning['best_model']
            best_params = lr_tuning['best_params']
        else:
            lr_model = self.models['Logistic Regression']
            best_params = "Default"

        lr_model.fit(X_train_pca, y_train)
        y_pred_lr = lr_model.predict(X_test_pca)

        cv_scores = cross_val_score(lr_model, X_train_pca, y_train, cv=cv, scoring='f1_weighted')

        results['Logistic Regression'] = {
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'f1_score': classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)['weighted avg']['f1-score'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
            'best_params': best_params,
            'features_used': 'PCA'
        }

        self.best_models['Logistic Regression'] = lr_model

        self.model_results = results
        return results

    def print_results(self):

        for model_name, result in self.model_results.items():
            print(f"\n{'='*50}")
            print(f"{model_name} Results (using {result['features_used']} features)")
            print(f"{'='*50}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"F1-Score: {result['f1_score']:.4f}")
            print(f"Cross-validation mean: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
            print(f"Best parameters: {result['best_params']}")
            print("\nClassification Report:")
            print(pd.DataFrame(result['classification_report']).transpose())
            print("\nConfusion Matrix:")
            print(result['confusion_matrix'])

    def get_best_model(self) -> Tuple[str, object]:

        best_model_name = max(self.model_results.keys(),
                             key=lambda k: self.model_results[k]['f1_score'])
        return best_model_name, self.best_models[best_model_name]

    def plot_feature_importance(self, model_name: str):

        if model_name in ['Random Forest']:
            model = self.best_models[model_name]
            if hasattr(model, 'feature_importances_'):

                feature_names = self.feature_processor.get_feature_names()

                importances = model.feature_importances_

                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(12, 6))
                plt.title(f"Top 20 Feature Importances - {model_name}")
                plt.bar(range(20), importances[indices[:20]], align="center")
                plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
                plt.tight_layout()
                plt.show()

class Pipeline:


    def __init__(self):
        self.downloader = BookDownloader()
        self.extractor = PDFTextExtractor()
        self.preprocessor = TextPreprocessor()
        self.analyzer = DataAnalyzer()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.ml_trainer = MLModelTrainer()
        self.book_paths = {}
        self.processed_data = []
        self.processed_texts = []
        self.metadatas = []
        self.book_labels = []

    def download_books(self) -> None:

        print("Downloading books...")
        self.book_paths = self.downloader.download_books()
        print(f"Downloaded {len(self.book_paths)} books")

    def extract_text(self) -> None:

        print("Extracting text from books...")
        self.processed_data = self.extractor.extract_all_books(self.book_paths)
        print(f"Extracted text from {len(self.processed_data)} pages")

    def analyze_data(self) -> Dict:

        print("Performing exploratory data analysis...")

        cleaned_texts = []
        for data in tqdm(self.processed_data, desc="Cleaning texts for analysis"):
            preprocessed = self.preprocessor.preprocess_text(data['text'])
            cleaned_texts.append(preprocessed['cleaned_text'])

        analysis_results = self.analyzer.analyze_text_data(cleaned_texts)

        print("\nText Analysis Summary:")
        print(f"Vocabulary size: {analysis_results['vocab_size']}")
        print(f"Average word count per text: {analysis_results['word_count_stats']['mean']:.2f}")
        print(f"Average sentence count per text: {analysis_results['sentence_count_stats']['mean']:.2f}")


        self.analyzer.plot_word_frequency(analysis_results['word_freq'])

        self.analyzer.plot_text_length_distribution(cleaned_texts)

        return analysis_results

    def preprocess_data(self) -> None:
        """Preprocess all extracted text"""
        print("Preprocessing text...")
        self.processed_texts = []
        self.metadatas = []
        self.book_labels = []

        for data in tqdm(self.processed_data, desc="Preprocessing"):

            preprocessed = self.preprocessor.preprocess_text(data['text'])

            chunks = self.embedding_generator.chunk_text(
                preprocessed['cleaned_text'], overlap=100
            )

            for chunk in chunks:
                self.processed_texts.append(chunk)
                self.metadatas.append({
                    'source': data['source'],
                    'page': data['page'],
                    'original_text': data['text']
                })

                self.book_labels.append(data['source'].replace('.pdf', ''))

        print(f"Created {len(self.processed_texts)} text chunks")
        print(f"Unique labels: {set(self.book_labels)}")

    def generate_and_store_embeddings(self) -> None:

        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(self.processed_texts)

        ids = [f"doc_{i}" for i in range(len(self.processed_texts))]

        print("Storing embeddings in vector database...")
        self.vector_db.add_documents(self.processed_texts, embeddings, self.metadatas, ids)
        print(f"Stored {len(self.processed_texts)} document chunks in vector database")

    def train_ml_models(self, tune_hyperparameters: bool = True) -> Dict:
        """Train and evaluate ML models"""
        print("Training ML models...")

        results = self.ml_trainer.train_and_evaluate(
            self.processed_texts, self.book_labels, tune_hyperparameters
        )

        self.ml_trainer.print_results()

        best_model_name, best_model = self.ml_trainer.get_best_model()
        print(f"\nBest performing model: {best_model_name}")

        if best_model_name in ['Random Forest']:
            self.ml_trainer.plot_feature_importance(best_model_name)

        return results

    def run_full_pipeline(self, perform_analysis: bool = True,
                         train_models: bool = True,
                         tune_hyperparameters: bool = True) -> Dict:

        pipeline_results = {}

        self.download_books()

        self.extract_text()

        if perform_analysis:
            analysis_results = self.analyze_data()
            pipeline_results['analysis'] = analysis_results

        self.preprocess_data()

        self.generate_and_store_embeddings()

        if train_models:
            model_results = self.train_ml_models(tune_hyperparameters)
            pipeline_results['models'] = model_results

        return pipeline_results

    def answer_question(self, query: str, n_results: int = TOP_K_RESULTS) -> str:
        preprocessed_query = self.preprocessor.preprocess_text(query)

        query_text = ' '.join(preprocessed_query['final_tokens'])

        query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]

        results = self.vector_db.query(query_embedding, n_results=n_results)

        answer = f"Based on the provided books, here's information related to your query '{query}':\n\n"

        for i in range(len(results['documents'][0])):
            source = results['metadatas'][0][i]['source']
            page = results['metadatas'][0][i]['page']
            text = results['documents'][0][i][:300] + "..."

            answer += f"From {source}, page {page}:\n{text}\n\n"

        return answer

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline_results = pipeline.run_full_pipeline(
        perform_analysis=True,
        train_models=True,
        tune_hyperparameters=True
    )
