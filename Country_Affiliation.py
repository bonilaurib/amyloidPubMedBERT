import pandas as pd
import re
import plotly.express as px

# List of countries and abbreviation mappings
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas",
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
    "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil",
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cape Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Democratic Republic of the Congo", "Denmark",
    "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Korea", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman",
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea",
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino", "São Tomé and Príncipe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone",
    "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan",
    "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe"
]

# Mapping of aliases for specific countries
country_aliases = {
    "US": "United States",
    "USA": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "BR": "Brazil",
    "Brasil": "Brazil",
    "RU": "Russia",
    "Russian Federation": "Russia"
}

# Abbreviations of US states
us_states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", 
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", 
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", 
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# List of major US institutions
us_institutions = [
    # University of California System
    "University of California", "UC Berkeley", "UC Davis", "UC Irvine", "UC Los Angeles", 
    "UC Riverside", "UC San Diego", "UC Santa Barbara", "UC Santa Cruz", "UCLA",

    # Ivy League Schools
    "Harvard University", "Yale University", "Princeton University", 
    "Columbia University", "University of Pennsylvania", "Dartmouth College", 
    "Brown University", "Cornell University",

    # Major Research Universities
    "Massachusetts Institute of Technology", "MIT", "Stanford University", 
    "California Institute of Technology", "Caltech", 
    "University of Chicago", "University of Michigan", 
    "University of Wisconsin-Madison", "University of Illinois Urbana-Champaign", 
    "Johns Hopkins University", "University of Washington", "University of Texas at Austin",

    # Prominent Private Universities
    "Duke University", "Northwestern University", "Emory University", 
    "University of Southern California", "USC", "Vanderbilt University", 
    "Rice University", "Washington University in St. Louis",

    # Medical Schools and Research Institutions
    "Mayo Clinic", "Cleveland Clinic", "MD Anderson Cancer Center", 
    "National Institutes of Health", "NIH", "Fred Hutchinson Cancer Research Center", 
    "Scripps Research Institute", "Broad Institute", "Howard Hughes Medical Institute",

    # National Labs and Agencies
    "Los Alamos National Laboratory", "Lawrence Berkeley National Laboratory", 
    "Argonne National Laboratory", "Oak Ridge National Laboratory", 
    "Sandia National Laboratories", "Jet Propulsion Laboratory", "JPL"
]

# Function to normalize country names
def normalize_country(affiliation):
    if affiliation == "No Affiliation":
        return None
    affiliation_lower = affiliation.lower()
    
    # Check for full country names first
    for country in countries:
        if country.lower() in affiliation_lower:
            return country

    # Check for aliases
    for alias, country in country_aliases.items():
        if alias.lower() in affiliation_lower:
            return country
    
    # Check for US institutions
    for institution in us_institutions:
        if institution.lower() in affiliation_lower:
            return "United States"
    
    # Check for US states
    for state in us_states:
        if f", {state.lower()} " in affiliation_lower or f", {state.lower()}." in affiliation_lower:
            return "United States"
    
    # Default to Unknown
    return "Unknown"

# Read the CSV
df = pd.read_csv("metadata.csv")

# Remove rows that contain only "No Affiliation"
df = df[~df['Affiliations'].str.fullmatch(r'(No Affiliation;?\s?)+')].copy()

# Add the "Country" column applying normalization
df['Country'] = df['Affiliations'].apply(normalize_country)

# Count the frequency of each country
country_counts = df['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Save the result to a CSV file
country_counts.to_csv("country_frequency.csv", index=False)

# Create the world map with country frequencies
fig = px.choropleth(country_counts, locations="Country", locationmode="country names",
                    color="Count", hover_name="Country", title="Country Frequency Map")
fig.update_layout(showlegend=False)
fig.show()

print("Processing complete!")
