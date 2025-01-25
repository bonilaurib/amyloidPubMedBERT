from Bio import Entrez
import time
import xml.etree.ElementTree as ET
import re

# Entrez configuration
Entrez.email = "@.com"  

# Function to clean text
def clean_text(text):
    if text:
        return text.strip().replace('"', '').replace("'", "")
    return "No Data"

# Function to extract full text
def extract_full_text(node):
    if node is None:
        return "No Data"
    return "".join(node.itertext()).strip()

# Function to remove forbidden words from text
def remove_forbidden_words(text, forbidden_words):
    for word in forbidden_words:
        pattern = r'(<.*?>)*\s*' + re.escape(word) + r'\s*(</.*?>)*'
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text.strip()

# Function to process XML returned by the API
def parse_abstracts(xml_data, forbidden_words):
    root = ET.fromstring(xml_data)
    abstracts = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else "Unknown PMID"
        title = article.find(".//ArticleTitle")
        title = extract_full_text(title) if title is not None else "No Title"
        
        authors = article.findall(".//Author")
        author_names = []
        for author in authors:
            last_name = author.find(".//LastName")
            fore_name = author.find(".//ForeName")
            name = f"{clean_text(last_name.text)}, {clean_text(fore_name.text)}" if last_name and fore_name else "No Name"
            author_names.append(name)
        
        journal = article.find(".//Journal/Title")
        journal = clean_text(journal.text) if journal is not None else "No Journal"
        
        pub_date = article.find(".//PubDate/Year")
        pub_date = clean_text(pub_date.text) if pub_date is not None else "No Date"
        
        abstract_texts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join(extract_full_text(a) for a in abstract_texts if a.text).strip() if abstract_texts else "No Abstract"
        
        if abstract != "No Abstract":
            abstract = remove_forbidden_words(abstract, forbidden_words)

        abstracts.append({
            "pmid": pmid,
            "title": title,
            "authors": "; ".join(author_names) if author_names else "No Authors",
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date
        })
    return abstracts

# Load PMIDs
with open("pmids.txt", "r") as f:
    pmid_list = [line.strip() for line in f if line.strip().isdigit()]

print(f"Total PMIDs loaded: {len(pmid_list)}")

# Forbidden words
forbidden_words = [
    "BACKGROUND:", "METHODS:", "OBJECTIVES:", "RESULTS:", "[Figurre: see text]", "•", "<strong>BACKGROUND</strong>",
    "CONCLUSIONS:", "CONCLUSION:", "INTRODUCTION:", "Conclusion:", "[Image: see text]",
    "Introduction:", "Methods:", "Results:", "Conclusions:", "Objectives:","[reaction: see text]"
]

all_abstracts = []
missing_abstracts = []

# Request in batches with retries
batch_size = 50
for i in range(0, len(pmid_list), batch_size):
    id_list = pmid_list[i:i + batch_size]
    id_str = ",".join(id_list)
    
    for attempt in range(3):  # Up to 3 attempts per batch
        try:
            print(f"Fetching abstracts for batch {i // batch_size + 1}, attempt {attempt + 1}")
            fetch_handle = Entrez.efetch(db="pubmed", id=id_str, rettype="abstract", retmode="xml")
            xml_data = fetch_handle.read()
            fetch_handle.close()
            
            if not xml_data.strip():  # If the returned XML is empty
                raise ValueError("Empty XML returned by the API.")
            
            with open(f"debug_batch_{i}.xml", "w") as xml_file:
                xml_file.write(xml_data)

            parsed_data = parse_abstracts(xml_data, forbidden_words)
            all_abstracts.extend(parsed_data)
            break  # Exit retry loop if successful
        except Exception as e:
            print(f"Error fetching batch {i // batch_size + 1}, attempt {attempt + 1}: {e}")
            time.sleep(5)  # Wait before retrying
    else:
        print(f"Failed all attempts for batch {i // batch_size + 1}.")
        missing_abstracts.extend(id_list)

    time.sleep(3)  # Respect API limits

# Save abstracts
with open("abstracts.txt", "w") as f:
    for abstract in all_abstracts:
        f.write(f"Abstract #{abstract['pmid']}\n")
        f.write(f"Title: {abstract['title']}\n")
        f.write(f"Authors: {abstract['authors']}\n")
        f.write(f"Abstract: {abstract['abstract']}\n")
        f.write(f"Journal: {abstract['journal']}\n")
        f.write(f"Publication Date: {abstract['pub_date']}\n")
        f.write("\n")

print("Abstracts successfully saved.")

# Save missing PMIDs
with open("missing_abstracts.txt", "w") as f:
    for pmid in missing_abstracts:
        f.write(f"{pmid}\n")

print(f"Total missing PMIDs: {len(missing_abstracts)}")
