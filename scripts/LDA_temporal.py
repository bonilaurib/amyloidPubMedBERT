import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Function to download NLTK resources if necessary
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' resource...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading 'stopwords' resource...")
        nltk.download('stopwords')

# Call the function to ensure resources are available
download_nltk_resources()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())  # Convert to lowercase
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Load data
data = pd.read_csv('metadata.csv')

# Extracting the year from the 'Publication_Date' column (assuming format 'YYYY-MM-DD' or similar)
data['Year'] = pd.to_datetime(data['Publication_Date'], errors='coerce').dt.year

# Filter data for the period 2020 to 2025
data = data[(data['Year'] >= 2020) & (data['Year'] <= 2025)]

# Apply preprocessing
data['tokens'] = data['Abstract'].apply(preprocess)

# Join tokens back into strings
data['processed_text'] = data['tokens'].apply(lambda x: ' '.join(x))

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['processed_text'])

# Define and train the LDA model
n_topics = 5  # Number of topics
lda_model = LDA(n_components=n_topics, random_state=42)
lda_model.fit(X)

# Compute topic distribution for each document
topic_distribution = lda_model.transform(X)

# Compute the average frequency of each topic
topic_frequencies = np.mean(topic_distribution, axis=0)

# Create a DataFrame for topics and their frequencies
topic_frequencies_df = pd.DataFrame({
    'Topic': [f'Topic {i + 1}' for i in range(n_topics)],
    'Frequency': topic_frequencies
})

# Retrieve the most important words for each topic
num_words = 50  # Number of words to display per topic
topic_words = {}

for index, topic in enumerate(lda_model.components_):
    words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
    topic_words[f'Topic {index + 1}'] = words

# Add most important words to the DataFrame
for i, words in enumerate(topic_words.values()):
    topic_frequencies_df.loc[i, 'Top_Words'] = ', '.join(words)

# Save as CSV for use in R
topic_frequencies_df.to_csv('topic_frequencies.csv', index=False)

# Display the DataFrame
print(topic_frequencies_df)
