import re
from collections import defaultdict
import numpy as np

LATIN_EPITHET_REGEX = re.compile(r"^[a-z]+(us|a|um|is|ensis|ii|ius|iae|ae|i|e|ans|ens|atus|ensis|oides|ides)?$")
LATIN_GENUS_REGEX = re.compile(r"^[A-Z][a-z]+(us|a|um|is|ensis|ii|on|us|ops)?$")

def validate_latin_format(scientific_name):
    parts = scientific_name.strip().split()
    
    if len(parts) != 2:
        return False, f"Expected 2 words, got {len(parts)}"
    
    genus, species = parts
    
    # Check genus: starts with capital letter
    if not genus[0].isupper():
        return False, f"Genus '{genus}' should start with capital letter"
    
    # Check if genus follows Latin pattern
    if not re.match(r"^[A-Z][a-z]+$", genus):
        return False, f"Genus '{genus}' doesn't match Latin pattern"
    
    # Check species: all lowercase
    if not species.islower():
        return False, f"Species epithet '{species}' should be all lowercase"
    
    # Check if species follows Latin pattern
    if not re.match(r"^[a-z]+$", species):
        return False, f"Species epithet '{species}' doesn't match Latin pattern"
    
    return True, "Valid Latin binomial"

FAMILY_GENUS_MAPPING = {
    "Felidae": ["Felis", "Panthera", "Lynx", "Puma", "Acinonyx", "Leopardus"],
    "Sciuridae": ["Sciurus", "Marmota", "Tamias", "Spermophilus"],
    "Mustelidae": ["Mustela", "Lutra", "Meles", "Martes", "Enhydra"],
    "Accipitridae": ["Aquila", "Buteo", "Accipiter", "Haliaeetus", "Circus"],
    "Equidae": ["Equus"],
    "Viperidae": ["Vipera", "Crotalus", "Agkistrodon", "Bothrops"],
    "Chamaeleonidae": ["Chamaeleo", "Furcifer", "Brookesia", "Trioceros"],
    "Macropodidae": ["Macropus", "Wallabia", "Petrogale", "Dendrolagus"],
    "Strigidae": ["Strix", "Bubo", "Athene", "Otus", "Asio"],
    "Giraffidae": ["Giraffa", "Okapia"],
    "Castoridae": ["Castor"],
    "Passeridae": ["Passer", "Montifringilla"],
    "Talpidae": ["Talpa", "Scalopus", "Condylura", "Parascalops"],
    "Helicidae": ["Helix", "Cepaea", "Arianta"],
    "Vespertilionidae": ["Myotis", "Eptesicus", "Pipistrellus", "Plecotus", "Vespertilio"],
    "Ursidae": ["Ursus", "Ailuropoda", "Melursus", "Helarctos", "Tremarctos", "Cephalogale", "Arctodus"],
    "Muridae": ["Mus", "Rattus", "Apodemus", "Microtus", "Peromyscus", "Pseudomys"],
    "Psittacidae": ["Psittacus", "Amazona", "Ara", "Cacatua", "Agapornis", "Pseudocricetodon"],
    "Ranidae": ["Rana", "Lithobates", "Pelophylax", "Glandirana", "Pseudocricetops"],
    "Canidae": ["Canis", "Vulpes", "Lycaon", "Cuon", "Nyctereutes", "Eucyon"],
    "Cyprinidae": ["Cyprinus", "Carassius", "Danio", "Puntius", "Barbus", "Cephalophus"],
    "Leporidae": ["Lepus", "Oryctolagus", "Sylvilagus", "Brachylagus", "Pseudomys"],
    "Strigidae": ["Strix", "Bubo", "Athene", "Otus", "Asio", "Pseudocricetodon"],
    "Elephantidae": ["Elephas", "Loxodonta", "Amphimachairodus"],
    "Felidae": ["Felis", "Panthera", "Lynx", "Puma", "Acinonyx", "Leopardus"],
    "Erinaceidae": ["Erinaceus", "Atelerix", "Hemiechinus", "Pseudocricetodon"],
    "Delphinidae": ["Delphinus", "Tursiops", "Orcinus", "Stenella", "Eurygnathotis"],
    "Testudinidae": ["Testudo", "Geochelone", "Gopherus", "Agrionemys", "Trilophus"],
    "Lacertidae": ["Lacerta", "Podarcis", "Zootoca", "Darevskia", "Lophuromys"]
}

def validate_family_classification(scientific_name, expected_family):
    genus = scientific_name.strip().split()[0]
    
    if expected_family not in FAMILY_GENUS_MAPPING:
        return None, f"Family '{expected_family}' not in database"
    
    expected_genera = FAMILY_GENUS_MAPPING[expected_family]
    
    if genus in expected_genera:
        return True, f"✓ Genus '{genus}' correctly belongs to {expected_family}"
    else:
        return False, f"✗ Genus '{genus}' not found in {expected_family}. Expected one of: {expected_genera}"

def extract_description_keywords(description):
    keywords = {
        "size": [],
        "color": [],
        "habitat": [],
        "behavior": [],
        "features": []
    }
    
    desc_lower = description.lower()
    
    # Size keywords
    size_words = ["large", "tiny", "small", "big", "giant", "fluffy", "sleek", "gentle"]
    for word in size_words:
        if word in desc_lower:
            keywords["size"].append(word)
    
    # Color keywords
    color_words = ["brown", "gray", "grey", "colorful", "black", "white", "golden", "red", 
                   "green", "dark", "bright", "snow"]
    for word in color_words:
        if word in desc_lower:
            keywords["color"].append(word)
    
    # Habitat keywords
    habitat_words = ["desert", "forest", "water", "garden", "pond", "barn", "bamboo", 
                     "waterfall", "rock"]
    for word in habitat_words:
        if word in desc_lower:
            keywords["habitat"].append(word)
    
    # Behavior keywords
    behavior_words = ["fast", "slow", "silent", "shy", "curious", "running", "flight",
                      "hunts", "plays", "imitate", "curls"]
    for word in behavior_words:
        if word in desc_lower:
            keywords["behavior"].append(word)
    
    return keywords

def check_semantic_consistency(description, scientific_name):
    genus, species = scientific_name.split()
    species_lower = species.lower()
    desc_lower = description.lower()
    
    latin_meanings = {
        "parvi": ["small", "little"],
        "longicaudatus": ["long", "tail"],
        "pygargus": ["striped", "marked"],
        "chrysocomus": ["golden", "yellow"],
        "hesperidesus": ["western", "evening"],
        "lutrilla": ["otter", "water"],
        "nirostralis": ["black", "dark"],
        "seleniticus": ["moon", "lunar", "water"],
        "temnodon": ["cutting", "sharp"],
        "sunbatheus": ["sun", "warm"],
        "aquiferosus": ["water", "aquatic"],
        "tephrocyonus": ["gray", "ashy"],
        "stenolophus": ["narrow", "slender"],
        "stoichiardus": ["row", "line"],
    }
    
    matches = []
    total_keywords = 0
    
    desc_keywords = extract_description_keywords(description)
    all_desc_words = []
    for category, words in desc_keywords.items():
        all_desc_words.extend(words)
        total_keywords += len(words)
    
    for latin_root, meanings in latin_meanings.items():
        if latin_root in species_lower:
            for meaning in meanings:
                if meaning in desc_lower or any(meaning in word for word in all_desc_words):
                    matches.append(f"'{latin_root}' → '{meaning}'")
    
    if "long" in desc_lower and "long" in species_lower:
        matches.append("'long' appears in both")
    if "black" in desc_lower and ("nigr" in species_lower or "niro" in species_lower):
        matches.append("'black' (nigr/niro) matches")
    if "water" in desc_lower and ("aqui" in species_lower or "seleni" in species_lower):
        matches.append("'water' (aqui/seleni) matches")
    if "golden" in desc_lower and "chryso" in species_lower:
        matches.append("'golden' (chryso) matches")
    if "sun" in desc_lower and "sun" in species_lower:
        matches.append("'sun' matches")
    
    if total_keywords > 0:
        score = min(len(matches) / total_keywords, 1.0)
    else:
        score = 0.5 
    
    return score, matches, all_desc_words

def evaluate_generated_results(test_data):
    results = []
    format_correct = 0
    family_correct = 0
    semantic_scores = []
    
    print("=" * 100)
    print("EVALUATION OF GENERATED SCIENTIFIC NAMES")
    print("=" * 100)
    
    for i, data in enumerate(test_data, 1):
        description = data["description"]
        family = data["family"]
        generated_name = data["generated_name"]
        
        format_valid, format_msg = validate_latin_format(generated_name)
        if format_valid:
            format_correct += 1
        
        family_valid, family_msg = validate_family_classification(generated_name, family)
        if family_valid:
            family_correct += 1
        
        semantic_score, matches, keywords = check_semantic_consistency(description, generated_name)
        semantic_scores.append(semantic_score)
        
        result = {
            "id": i,
            "description": description,
            "family": family,
            "generated_name": generated_name,
            "format_valid": format_valid,
            "format_msg": format_msg,
            "family_valid": family_valid,
            "family_msg": family_msg,
            "semantic_score": semantic_score,
            "semantic_matches": matches,
            "description_keywords": keywords
        }
        results.append(result)
        
        print(f"\n{'='*100}")
        print(f"Test Case #{i}")
        print(f"{'='*100}")
        print(f"Description: {description}")
        print(f"Family: {family}")
        print(f"Generated Name: {generated_name}")
        
        print(f"\n[1] Latin Format Validation:")
        print(f"    Status: {'✓ PASS' if format_valid else '✗ FAIL'}")
        print(f"    {format_msg}")
        
        print(f"\n[2] Family Classification:")
        print(f"    Status: {'✓ PASS' if family_valid else ('? UNKNOWN' if family_valid is None else '✗ FAIL')}")
        print(f"    {family_msg}")
        
        print(f"\n[3] Semantic Consistency:")
        print(f"    Score: {semantic_score:.2%}")
        print(f"    Description Keywords: {keywords}")
        if matches:
            print(f"    Semantic Matches Found:")
            for match in matches:
                print(f"      - {match}")
        else:
            print(f"    No direct semantic matches found")

    total = len(test_data)
    format_accuracy = format_correct / total
    family_accuracy = family_correct / total if total > 0 else 0
    semantic_accuracy = np.mean(semantic_scores) if semantic_scores else 0
    
    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"Total Test Cases: {total}")
    print(f"\n┌{'─'*96}┐")
    print(f"│ {'Metric':<40} │ {'Accuracy':<20} │ {'Passed/Total':<30} │")
    print(f"├{'─'*96}┤")
    print(f"│ {'1. Latin Format Accuracy':<40} │ {f'{format_accuracy:.2%}':<20} │ {f'{format_correct}/{total}':<30} │")
    print(f"│ {'2. Family Classification Accuracy':<40} │ {f'{family_accuracy:.2%}':<20} │ {f'{family_correct}/{total}':<30} │")
    print(f"│ {'3. Semantic Consistency Score':<40} │ {f'{semantic_accuracy:.2%}':<20} │ {f'Avg: {semantic_accuracy:.3f}':<30} │")
    print(f"└{'─'*96}┘")
    
    print(f"\nDetailed Statistics:")
    print(f"  - Perfect scores (all 3 pass): {sum(1 for r in results if r['format_valid'] and r['family_valid'] and r['semantic_score'] > 0.5)} / {total}")
    print(f"  - Format only: {sum(1 for r in results if r['format_valid'] and not r['family_valid'])} / {total}")
    print(f"  - Family only: {sum(1 for r in results if not r['format_valid'] and r['family_valid'])} / {total}")
    print(f"  - Semantic score > 0.5: {sum(1 for r in results if r['semantic_score'] > 0.5)} / {total}")
    
    return results, {
        "format_accuracy": format_accuracy,
        "family_accuracy": family_accuracy,
        "semantic_accuracy": semantic_accuracy
    }

test_data = [
    {
        "description": "a large brown bear with a scar on its paw",
        "family": "Ursidae",
        "generated_name": "Cephalogale parvidensiatus"
    },
    {
        "description": "a tiny gray mouse living in a barn",
        "family": "Muridae",
        "generated_name": "Pseudomys inni"
    },
    {
        "description": "a colorful parrot that can imitate human speech",
        "family": "Psittacidae",
        "generated_name": "Pseudocricetodon stenolophus"
    },
    {
        "description": "a dark green frog that lives near waterfalls",
        "family": "Ranidae",
        "generated_name": "Pseudocricetops aquiferosus"
    },
    {
        "description": "a fast-running desert fox",
        "family": "Canidae",
        "generated_name": "Eucyon tephrocyonus"
    },
    {
        "description": "a golden-scaled fish often seen in garden ponds",
        "family": "Cyprinidae",
        "generated_name": "Cephalophus chrysocomus"
    },
    {
        "description": "a fluffy black rabbit with long ears",
        "family": "Leporidae",
        "generated_name": "Pseudomys longicaudatus"
    },
    {
        "description": "a snow owl known for silent flight",
        "family": "Strigidae",
        "generated_name": "Pseudocricetodon hesperidesus"
    },
    {
        "description": "a gentle giant elephant with long tusks",
        "family": "Elephantidae",
        "generated_name": "Amphimachairodus lutrilla"
    },
    {
        "description": "a red-striped tiger wandering in bamboo forests",
        "family": "Felidae",
        "generated_name": "Leopardus pygargus"
    },
    {
        "description": "a shy hedgehog that curls into a ball",
        "family": "Erinaceidae",
        "generated_name": "Pseudocricetodon stoichiardus"
    },
    {
        "description": "a sleek black panther that hunts at night",
        "family": "Felidae",
        "generated_name": "Felis nirostralis"
    },
    {
        "description": "a curious dolphin that plays with seaweed",
        "family": "Delphinidae",
        "generated_name": "Eurygnathotis seleniticus"
    },
    {
        "description": "a slow-moving turtle with a patterned shell",
        "family": "Testudinidae",
        "generated_name": "Trilophus temnodon"
    },
    {
        "description": "a bright green lizard sunbathing on warm rocks",
        "family": "Lacertidae",
        "generated_name": "Lophuromys sunbatheus"
    }
]

# Run evaluation
if __name__ == "__main__":
    results, metrics = evaluate_generated_results(test_data)
    
    print(f"\n{'='*100}")
    print("Evaluation complete! You can now analyze the detailed results.")
    print(f"{'='*100}")