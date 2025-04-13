import pandas as pd
import numpy as np
import re
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datasets import load_dataset
import os

# Custom transformer to combine embeddings and structured features
class EmbeddingFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, embedder, scaler=None):
        self.embedder = embedder
        self.scaler = scaler
        # Store the model name during initialization
        self.embedder_name = embedder._model_name if hasattr(embedder, '_model_name') else "all-MiniLM-L6-v2"

    def fit(self, X, y=None):
        # Fit the scaler on the structured features
        self.scaler.fit(X[['question_length', 'num_numbers', 'num_steps']].values)
        return self

    def __getstate__(self):
        """Custom serialization method"""
        state = self.__dict__.copy()
        # Save just the model name
        state['embedder_name'] = self.embedder_name
        state.pop('embedder')
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__.update(state)
        # Restore the embedder from the saved name
        self.embedder = SentenceTransformer(self.embedder_name)
        
    def transform(self, X):
        # Get text embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = self.embedder.encode(X['question'].tolist(), show_progress_bar=True, device=device)

        # Scale structured features
        scaled_features = self.scaler.transform(X[['question_length', 'num_numbers', 'num_steps']].values)

        # Combine both embeddings and scaled features
        return np.hstack([embeddings, scaled_features])

# Function to extract features from the dataset
def extract_features(example):
    q = example["question"]
    a = example["answer"]

    return {
        "question": q,
        "answer": a,
        "question_length": len(q.split()),
        "num_numbers": len(re.findall(r'\d+', q)),
        "num_steps": len(re.findall(r'<<.*?>>', a)) if "<<" in a else a.count("\n"),  # each step enclosed like <<...>> or count newlines
        "final_answer": a.split("####")[-1].strip() if "####" in a else a
    }

def create_clustering_model(num_clusters=5, output_dir=None):
    """Create and save the clustering model with proper path handling and error checking"""
    try:
        print("Creating GSM8K question clustering model...")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths
        output_path = os.path.join(output_dir, "gsm8k_cluster_pipeline.pkl")
        stats_path = os.path.join(output_dir, "gsm8k_cluster_stats.csv")
        questions_path = os.path.join(output_dir, "gsm8k_clustered_questions.csv")
        
        print(f"Will save outputs to:")
        print(f"- Model: {output_path}")
        print(f"- Statistics: {stats_path}")
        print(f"- Questions: {questions_path}")
        
        # Load the GSM8K dataset
        print("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main")
        
        # Apply the feature extraction function
        df = pd.DataFrame([extract_features(ex) for ex in dataset["train"]])
        print(f"Loaded {len(df)} training examples")
        
        # Initialize the SentenceTransformer model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Scale the structured features
        scaler = StandardScaler()
        
        # Create embeddings and structured features
        print("Creating embeddings...")
        df["embedding"] = embedder.encode(df["question"].tolist(), show_progress_bar=True).tolist()
        
        # Select and scale structured features
        structured_features = df[["question_length", "num_numbers", "num_steps"]].values
        scaled_features = scaler.fit_transform(structured_features)
        
        # Combine with embeddings
        embedding_matrix = np.array(df["embedding"].tolist())
        X_combined = np.hstack([embedding_matrix, scaled_features])
        
        # Perform clustering
        print(f"Clustering data into {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_combined)
        
        # Print cluster examples
        for c in sorted(df["cluster"].unique()):
            print(f"\nCluster {c}:")
            examples = df[df["cluster"] == c]["question"].sample(min(2, len(df[df["cluster"] == c]))).values
            for example in examples:
                print(f"- {example[:100]}...")
        
        # Create pipeline
        print("\nBuilding clustering pipeline...")
        pipeline = Pipeline([
            ('embedding_feature_union', EmbeddingFeatureUnion(embedder, scaler)),
            ('kmeans', kmeans)
        ])
        
    
        # Save cluster statistics
        cluster_stats = df.groupby("cluster")[["question_length", "num_numbers", "num_steps"]].mean()
        print("\nCluster statistics:")
        print(cluster_stats)
        cluster_stats.to_csv(stats_path)
        
        # Save clustered questions
        df.to_csv(questions_path, index=False)
        
        print(f"\nOutputs saved to:")
        print(f"- Model: {output_path}")
        print(f"- Statistics: {stats_path}")
        print(f"- Questions: {questions_path}")
        
         # Save pipeline
        print(f"\nSaving model to {output_path}")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(pipeline, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
        
        return pipeline
        
    except Exception as e:
        print(f"Error creating clustering model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# Load the clustering pipeline
def load_cluster_pipeline(model_path):
    """Load the pretrained clustering model with robust error handling"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
            
        # Verify pipeline components
        if not hasattr(pipeline, 'named_steps'):
            raise ValueError("Loaded object is not a valid scikit-learn pipeline")
            
        if 'embedding_feature_union' not in pipeline.named_steps:
            raise ValueError("Pipeline is missing embedding_feature_union step")
            
        print(f"Successfully loaded clustering model from {model_path}")
        return pipeline
        
    except Exception as e:
        print(f"Error loading clustering model: {str(e)}")
        return None

def predict_cluster(pipeline, example):
    """Predict the cluster for a single example with all features."""
    features = extract_features(example)
    df = pd.DataFrame([features])
    return pipeline.predict(df)[0]

def analyze_clusters(samples):
    """
    Analyze results by cluster.
    
    Args:
        samples: List of sample dictionaries with results
        
    Returns:
        dict: Statistics about performance by cluster
    """
    import pandas as pd
    import numpy as np
    
    # Convert samples to dataframe for analysis
    data = []
    for sample in samples:
        # Skip samples without cluster information
        if "cluster" not in sample:
            continue
        
        # For samples with multiple predictions, use the first one
        is_correct = False
        if "pred" in sample and len(sample["pred"]) > 0:
            pred = sample["pred"][0]
            gt = sample["gt"]
            is_correct = pred == gt
        
        data.append({
            "idx": sample["idx"],
            "cluster": sample["cluster"],
            "is_correct": is_correct
        })
    
    # If no cluster data, return empty dict
    if not data:
        return {}
    
    # Create dataframe and analyze
    df = pd.DataFrame(data)
    
    # Calculate accuracy by cluster
    cluster_accuracy = df.groupby('cluster')['is_correct'].mean()
    cluster_counts = df.groupby('cluster').size()
    
    # Prepare results
    cluster_stats = {}
    for cluster, accuracy in cluster_accuracy.items():
        cluster_stats[str(cluster)] = {
            "accuracy": float(accuracy),
            "count": int(cluster_counts[cluster])
        }
    
    # Add overall accuracy
    overall_accuracy = df['is_correct'].mean()
    cluster_stats["overall"] = {
        "accuracy": float(overall_accuracy),
        "count": len(df)
    }
    
    return cluster_stats

if __name__ == "__main__":
    create_clustering_model()