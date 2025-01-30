import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load data
embeddings_path = "embeddings.csv"
metadata_path = "metadata.csv"

# Read files
embeddings = pd.read_csv(embeddings_path)
metadata = pd.read_csv(metadata_path)

# Initial diagnostics
print(f"Embeddings: {embeddings.shape}")
print(f"Metadata: {metadata.shape}")

# Ensure both datasets have a common column (e.g., ID)
if 'ID' not in embeddings.columns or 'ID' not in metadata.columns:
    raise ValueError("Both files must contain an 'ID' column for alignment.")

# Set 'ID' as the index
embeddings.set_index('ID', inplace=True)
metadata.set_index('ID', inplace=True)

# Align data using common indices
common_indices = embeddings.index.intersection(metadata.index)

# Diagnostics after intersection
print(f"Number of common indices: {len(common_indices)}")

if len(common_indices) == 0:
    raise ValueError("No common indices found between embeddings and metadata.")

# Filter data based on common indices
embeddings = embeddings.loc[common_indices]
metadata = metadata.loc[common_indices]

# Check or create the 'label' column
if 'label' not in metadata.columns:
    if 'Publication_Date' not in metadata.columns:
        raise ValueError("The 'Publication_Date' column is missing in metadata.")
    metadata['year'] = pd.to_datetime(metadata['Publication_Date'], errors='coerce').dt.year
    metadata.dropna(subset=['year'], inplace=True)
    metadata['decade'] = (metadata['year'] // 10) * 10
    metadata['label'] = metadata['decade']

# Ensure there are no invalid values in 'label'
metadata.dropna(subset=['label'], inplace=True)
metadata['label'] = metadata['label'].astype(str)  # Convert to string
y = metadata['label'].values

# Final dimension check
print(f"Final embeddings dimensions: {embeddings.shape}")
print(f"Final number of labels: {len(y)}")

if embeddings.shape[0] == 0 or len(y) == 0:
    raise ValueError("Data is empty after alignment. Check the input files.")

# Balance dataset using SMOTE
USE_SMOTE = True  # Flag to control SMOTE usage
if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(embeddings.values, y)
else:
    X_resampled, y_resampled = embeddings.values, y  # Use original dataset if SMOTE is disabled

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Create k-NN model
K_NEIGHBORS = min(5, len(X_train))  
model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, weights="distance", n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Display results
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"Weighted Accuracy: {accuracy:.2f}")
print(f"Weighted Precision: {precision:.2f}")
print(f"Weighted Recall: {recall:.2f}")
print(f"Weighted F1-Score: {f1:.2f}")

# Save results
results = [
    ["PubMedBERT", len(X_train), len(X_test), K_NEIGHBORS, accuracy, "SMOTE" if USE_SMOTE else "No Oversampling"]
]
results_df = pd.DataFrame(results, columns=["Model", "Training Size", "Test Size", "k", "Accuracy", "Mitigation"])
results_df.to_csv("knn_results_optimized.csv", index=False)

print("\n--- Results Table ---\n", results_df)
