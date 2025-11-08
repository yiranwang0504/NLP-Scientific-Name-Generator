import requests
import pandas as pd
import random
import time
import os

os.makedirs("data", exist_ok=True)
families = ["Canidae", "Felidae", "Ursidae", "Cervidae", "Bovidae",
            "Equidae", "Hominidae", "Muridae", "Sciuridae", "Crocodylidae"]

def get_gbif_key(name):
    """Get GBIF taxonKey for a given family name"""
    url = f"https://api.gbif.org/v1/species/match?name={name}"
    r = requests.get(url).json()
    return r.get("usageKey")

def safe_request(url, max_retries=5):
    for i in range(max_retries):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (GBIF Collector)"},
                timeout=30
            )
            if r.status_code == 200:
                return r
            else:
                print(f"⚠️ HTTP {r.status_code} — wait & retry ({i+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error {e} — retrying ({i+1}/{max_retries})")
        time.sleep(2 + random.random() * 3)
    print(" All retries failed:", url)
    return None

def get_children_recursive(taxon_key, level=0):
    names = []
    url = f"https://api.gbif.org/v1/species/{taxon_key}/children?limit=500"
    r = safe_request(url)
    if not r:
        return names

    data = r.json()
    for item in data.get("results", []):
        rank = item.get("rank")
        if rank == "SPECIES":
            names.append({
                "scientificName": item.get("scientificName"),
                "canonicalName": item.get("canonicalName"),
                "authorship": item.get("authorship"),
                "family": item.get("family"),
            })
        elif rank not in ["SPECIES", "SUBSPECIES"]:
            sub_key = item.get("key")
            time.sleep(0.5 + random.random()*0.3)
            names.extend(get_children_recursive(sub_key, level+1))
    return names


all_species = []
for fam in families:
    key = get_gbif_key(fam)
    print(f"{fam}: key={key}")
    data = get_children_recursive(key)
    all_species.extend(data)

df = pd.DataFrame(all_species)
# df_species = df[df["rank"] == "SPECIES"]
df.to_csv("data/species_list.csv", index=False)

print(df.head())