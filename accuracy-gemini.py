import re
from collections import defaultdict
import numpy as np

LATIN_EPITHET_REGEX = re.compile(r"^[a-z]+(us|a|um|is|ensis|ii|ius|iae|ae|i|e|ans|ens|atus|ensis|oides|ides|or|tor)?$")
LATIN_GENUS_REGEX = re.compile(r"^[A-Z][a-z]+$")

def validate_latin_format(scientific_name):
    scientific_name = scientific_name.replace('*', '').strip()
    
    parts = scientific_name.strip().split()
    
    if len(parts) != 2:
        return False, f"Expected 2 words, got {len(parts)}"
    
    genus, species = parts
    
    # Check genus: starts with capital letter
    if not genus[0].isupper():
        return False, f"Genus '{genus}' should start with capital letter"
    
    # Check if genus follows Latin pattern
    if not LATIN_GENUS_REGEX.match(genus):
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
    "Ursidae": ["Ursus", "Ailuropoda", "Melursus", "Helarctos", "Tremarctos"],
    "Muridae": ["Mus", "Rattus", "Apodemus", "Microtus", "Peromyscus"],
    "Psittacidae": ["Psittacus", "Amazona", "Ara", "Cacatua", "Agapornis"],
    "Ranidae": ["Rana", "Lithobates", "Pelophylax", "Glandirana"],
    "Canidae": ["Canis", "Vulpes", "Lycaon", "Cuon", "Nyctereutes"],
    "Cyprinidae": ["Cyprinus", "Carassius", "Danio", "Puntius", "Barbus"],
    "Leporidae": ["Lepus", "Oryctolagus", "Sylvilagus", "Brachylagus"],
    "Elephantidae": ["Elephas", "Loxodonta"],
    "Erinaceidae": ["Erinaceus", "Atelerix", "Hemiechinus"],
    "Delphinidae": ["Delphinus", "Tursiops", "Orcinus", "Stenella"],
    "Testudinidae": ["Testudo", "Geochelone", "Gopherus", "Agrionemys"],
    "Lacertidae": ["Lacerta", "Podarcis", "Zootoca", "Darevskia"]
}

def validate_family_classification(scientific_name, expected_family):
    scientific_name = scientific_name.replace('*', '').strip()
    genus = scientific_name.split()[0]
    
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
    size_words = ["large", "tiny", "small", "big", "giant", "fluffy", "sleek", "majestic", 
                  "long-necked", "powerful"]
    for word in size_words:
        if word in desc_lower:
            keywords["size"].append(word)
    
    # Color keywords
    color_words = ["brown", "gray", "grey", "colorful", "black", "white", "golden", "red", 
                   "green", "dark", "bright", "blue", "silvery", "striped"]
    for word in color_words:
        if word in desc_lower:
            keywords["color"].append(word)
    
    # Habitat keywords
    habitat_words = ["desert", "forest", "water", "garden", "pond", "barn", "bamboo", 
                     "waterfall", "rock", "savanna", "riverbank", "cave", "dam"]
    for word in habitat_words:
        if word in desc_lower:
            keywords["habitat"].append(word)
    
    # Behavior keywords
    behavior_words = ["fast", "slow", "silent", "shy", "curious", "running", "flight",
                      "hunts", "plays", "imitate", "curls", "agile", "soaring", "grazing",
                      "hopping", "hooting", "singing", "burrowing", "flying", "building",
                      "slides", "changes", "reaching"]
    for word in behavior_words:
        if word in desc_lower:
            keywords["behavior"].append(word)
    
    # Feature keywords
    feature_words = ["mane", "tail", "eyes", "eyesight", "claws", "feathers", "keen",
                     "venomous", "rattling", "round", "leaves", "nuts"]
    for word in feature_words:
        if word in desc_lower:
            keywords["features"].append(word)
    
    return keywords


def check_semantic_consistency(description, scientific_name):
    scientific_name = scientific_name.replace('*', '').strip()
    genus, species = scientific_name.split()
    species_lower = species.lower()
    desc_lower = description.lower()
    
    latin_meanings = {
        "crinita": ["mane", "hair", "flowing"],
        "nucifraga": ["nut", "gather"],
        "ludicra": ["playful", "play", "game"],
        "acuta": ["sharp", "keen", "acute"],
        "vittatus": ["striped", "banded"],
        "sonans": ["sound", "rattling", "noise"],
        "versicolor": ["color", "changing", "varied"],
        "saltator": ["jumping", "hopping", "leap"],
        "oculata": ["eye", "eyes", "vision"],
        "alta": ["tall", "high", "long"],
        "aedificans": ["building", "construct"],
        "caeruleus": ["blue", "azure"],
        "fossor": ["digging", "burrowing"],
        "argenteus": ["silver", "silvery"],
        "speluncae": ["cave", "cavern"],
        "parvi": ["small", "little"],
        "longicaudatus": ["long", "tail"],
        "pygargus": ["striped", "marked"],
        "chrysocomus": ["golden", "yellow"],
        "aquiferosus": ["water", "aquatic"],
        "tephrocyonus": ["gray", "ashy"],
    }
    
    matches = []
    
    desc_keywords = extract_description_keywords(description)
    all_desc_words = []
    for category, words in desc_keywords.items():
        all_desc_words.extend(words)
    
    for latin_root, meanings in latin_meanings.items():
        if latin_root in species_lower:
            for meaning in meanings:
                if meaning in desc_lower:
                    matches.append(f"'{latin_root}' → '{meaning}' ✓")
                elif any(meaning in word for word in all_desc_words):
                    matches.append(f"'{latin_root}' → '{meaning}' (partial) ✓")
    
    direct_matches = {
        "mane": "crinita",
        "nut": "nucifraga",
        "play": "ludicra",
        "keen": "acuta",
        "sharp": "acuta",
        "stripe": "vittatus",
        "sound": "sonans",
        "rattle": "sonans",
        "color": "versicolor",
        "change": "versicolor",
        "jump": "saltator",
        "hop": "saltator",
        "eye": "oculata",
        "tall": "alta",
        "high": "alta",
        "long": "alta",
        "build": "aedificans",
        "blue": "caeruleus",
        "dig": "fossor",
        "burrow": "fossor",
        "silver": "argenteus",
        "cave": "speluncae",
    }
    
    for desc_word, latin_word in direct_matches.items():
        if desc_word in desc_lower and latin_word in species_lower:
            if not any(latin_word in m for m in matches):
                matches.append(f"'{desc_word}' → '{latin_word}' (direct) ✓")
    
    total_keywords = len(all_desc_words)
    if total_keywords > 0:
        if len(matches) > 0:
            score = min(0.5 + (len(matches) * 0.25), 1.0)
        else:
            score = 0.0
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
        generated_name = data["generated_name"].replace('*', '').strip()
        
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
    print(f"  - Format + Family correct: {sum(1 for r in results if r['format_valid'] and r['family_valid'])} / {total}")
    print(f"  - Format only: {sum(1 for r in results if r['format_valid'] and not r['family_valid'])} / {total}")
    print(f"  - Family only: {sum(1 for r in results if not r['format_valid'] and r['family_valid'])} / {total}")
    print(f"  - Semantic score ≥ 0.5: {sum(1 for r in results if r['semantic_score'] >= 0.5)} / {total}")
    print(f"  - Semantic score ≥ 0.75: {sum(1 for r in results if r['semantic_score'] >= 0.75)} / {total}")
    
    return results, {
        "format_accuracy": format_accuracy,
        "family_accuracy": family_accuracy,
        "semantic_accuracy": semantic_accuracy
    }

test_data = [
    {
        "description": "a majestic lion with a flowing mane",
        "family": "Felidae",
        "generated_name": "*Panthera crinita*"
    },
    {
        "description": "a small, agile squirrel that gathers nuts",
        "family": "Sciuridae",
        "generated_name": "*Sciurus nucifraga*"
    },
    {
        "description": "a playful otter that slides on riverbanks",
        "family": "Mustelidae",
        "generated_name": "*Lutra ludicra*"
    },
    {
        "description": "a soaring eagle with keen eyesight",
        "family": "Accipitridae",
        "generated_name": "*Aquila acuta*"
    },
    {
        "description": "a striped zebra grazing on the savanna",
        "family": "Equidae",
        "generated_name": "*Equus vittatus*"
    },
    {
        "description": "a venomous snake with a rattling tail",
        "family": "Viperidae",
        "generated_name": "*Vipera sonans*"
    },
    {
        "description": "a colorful chameleon that changes skin hue",
        "family": "Chamaeleonidae",
        "generated_name": "*Chamaeleo versicolor*"
    },
    {
        "description": "a hopping kangaroo with a powerful tail",
        "family": "Macropodidae",
        "generated_name": "*Macropus saltator*"
    },
    {
        "description": "a hooting owl with large, round eyes",
        "family": "Strigidae",
        "generated_name": "*Strix oculata*"
    },
    {
        "description": "a long-necked giraffe reaching for leaves",
        "family": "Giraffidae",
        "generated_name": "*Giraffa alta*"
    },
    {
        "description": "a busy beaver building a dam",
        "family": "Castoridae",
        "generated_name": "*Castor aedificans*"
    },
    {
        "description": "a singing bird with bright blue feathers",
        "family": "Passeridae",
        "generated_name": "*Passer caeruleus*"
    },
    {
        "description": "a burrowing mole with strong claws",
        "family": "Talpidae",
        "generated_name": "*Talpa fossor*"
    },
    {
        "description": "a slow-moving snail leaving a silvery trail",
        "family": "Helicidae",
        "generated_name": "*Helix argenteus*"
    },
    {
        "description": "a nocturnal bat flying in caves",
        "family": "Vespertilionidae",
        "generated_name": "*Myotis speluncae*"
    }
]

# Run evaluation
if __name__ == "__main__":
    results, metrics = evaluate_generated_results(test_data)
    
    print(f"\n{'='*100}")
    print("Evaluation complete! You can now analyze the detailed results.")
    print(f"{'='*100}")