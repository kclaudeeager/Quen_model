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

# Custom transformer to combine embeddings and structured features
class EmbeddingFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, embedder, scaler=None):
        self.embedder = embedder
        self.scaler = scaler

    def fit(self, X, y=None):
        # Fit the scaler on the structured features
        self.scaler.fit(X[['question_length', 'num_numbers', 'num_steps']].values)
        return self

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

def create_clustering_model(num_clusters=5):
    print("Creating GSM8K question clustering model...")
    
    # Load the GSM8K dataset
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
    
    # Select structured features
    structured_features = df[["question_length", "num_numbers", "num_steps"]].values
    scaled_features = scaler.fit_transform(structured_features)
    
    # Combine with embeddings
    embedding_matrix = np.array(df["embedding"].tolist())
    X_combined = np.hstack([embedding_matrix, scaled_features])
    
    # Perform clustering
    print(f"Clustering data into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_combined)
    
    # Get examples from each cluster
    for c in sorted(df["cluster"].unique()):
        print(f"\nCluster {c}:")
        examples = df[df["cluster"] == c]["question"].sample(2).values
        for example in examples:
            print(f"- {example[:100]}...")
    
    # Create a pipeline that can predict clusters for new questions
    print("\nBuilding clustering pipeline...")
    pipeline = Pipeline([
        ('embedding_feature_union', EmbeddingFeatureUnion(embedder, scaler)),
        ('kmeans', kmeans)
    ])
    
    # Save the pipeline
    output_path = "gsm8k_cluster_pipeline.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Clustering model saved to {output_path}")
    
    # Save cluster statistics
    cluster_stats = df.groupby("cluster")[["question_length", "num_numbers", "num_steps"]].mean()
    print("\nCluster statistics:")
    print(cluster_stats)
    
    cluster_stats.to_csv("gsm8k_cluster_stats.csv")
    
    # Save clustered questions
    df.to_csv("gsm8k_clustered_questions.csv", index=False)
    
    return pipeline

# Load the clustering pipeline
def load_cluster_pipeline(model_path="gsm8k_cluster_pipeline.pkl"):
    """Load the pretrained clustering model."""
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Clustering model loaded from {model_path}")
        return pipeline
    except FileNotFoundError:
        print(f"Error: Clustering model not found at {model_path}")
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