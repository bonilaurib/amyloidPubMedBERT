import pandas as pd
import plotly.express as px
from collections import Counter
import json
import os
import pubchempy as pcp
from chemdataextractor import Document
from tqdm import tqdm
import logging

# Logging configuration
logging.basicConfig(
    filename='error_log.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data cache to avoid repeated API calls
api_cache_file = 'api_cache.json'
if os.path.exists(api_cache_file):
    with open(api_cache_file, 'r') as f:
        api_cache = json.load(f)
else:
    api_cache = {}

# Function to save the cache
def save_cache():
    with open(api_cache_file, 'w') as f:
        json.dump(api_cache, f)

# Function to retrieve PubChem information
def get_pubchem_info(compound_name):
    if compound_name in api_cache:
        return api_cache[compound_name]
    
    retries = 3
    for attempt in range(retries):
        try:
            compounds = pcp.get_compounds(compound_name, 'name')
            if compounds:
                compound = compounds[0]
                pubchem_info = {
                    'molecular_formula': compound.molecular_formula,
                    'iupac_name': compound.iupac_name
                }
                api_cache[compound_name] = pubchem_info
                save_cache()
                return pubchem_info
            else:
                logging.error(f"Compound {compound_name} not found in PubChem.")
                return {'molecular_formula': 'N/A', 'iupac_name': 'N/A'}
        except Exception as e:
            logging.error(f"Error retrieving PubChem data for {compound_name}: {e}")
    return {'molecular_formula': 'N/A', 'iupac_name': 'N/A'}

# Function to extract compounds from abstracts
def extract_compounds(abstracts):
    compounds = []
    for abstract in abstracts:
        doc = Document(abstract)
        for chem in doc.cems:
            name = chem.text.strip().lower()
            pubchem_info = get_pubchem_info(name)
            chebi_id = 'N/A'  # ChEBI integration can be added later
            compounds.append((name, chebi_id, pubchem_info))
    return compounds

# Function to map ChEBI class IDs to names
def map_chebi_class_to_name(chebi_class_id):
    chebi_class_names = {
        "CHEBI:23432": "Non-steroidal anti-inflammatory drug",
        "CHEBI:23431": "Analgesic",
        "CHEBI:23638": "Xanthine alkaloid",
        "CHEBI:23400": "Anti-inflammatory drug",
    }
    return chebi_class_names.get(chebi_class_id, "Unknown Class")

# Load the CSV file containing the "Abstract" column
try:
    df = pd.read_csv("metadata.csv")
except FileNotFoundError:
    logging.error("CSV file not found. Ensure 'metadata70_80.csv' is in the current directory.")
    raise

# Check if "Abstract" column exists
if "Abstract" not in df.columns:
    logging.error("'Abstract' column missing in the dataset.")
    raise KeyError("'Abstract' column not found in the dataset.")

# Apply the extraction function with a progress bar
tqdm.pandas()
df["Extracted Compounds"] = df["Abstract"].progress_apply(lambda x: extract_compounds([x]))

# Prepare a DataFrame for compounds
compounds_data = []

for index, row in df.iterrows():
    for compound, chebi_id, pubchem_info in row["Extracted Compounds"]:
        chebi_class_name = map_chebi_class_to_name(chebi_id)
        compounds_data.append({
            "Term": compound,
            "Entity Type": "Chemical",
            "ChEBI Class": chebi_id,
            "PubChem Info": json.dumps(pubchem_info),  # Serialize to handle dictionaries
            "Frequency": 1,
            "ChEBI Class Name": chebi_class_name
        })

# Create a DataFrame for the compounds
compounds_df = pd.DataFrame(compounds_data)

# Aggregate frequency
compounds_df = compounds_df.groupby(
    ["Term", "Entity Type", "ChEBI Class", "ChEBI Class Name"], as_index=False
).agg({
    "PubChem Info": "first",
    "Frequency": "sum"
})

# Generate a treemap using Plotly
fig = px.treemap(
    compounds_df,
    path=["Term", "ChEBI Class Name"],
    values="Frequency",
    color="Frequency",
    hover_data={"PubChem Info": True},
    title="Chemical Compounds Treemap"
)

# Display the treemap
fig.show()

# Optionally save the treemap as an image
fig.write_image("treemap.png")

# Save the compounds DataFrame as a CSV
compounds_df.to_csv("compounds_summary.csv", index=False)
