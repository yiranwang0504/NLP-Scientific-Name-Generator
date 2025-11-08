import os
import time, random, json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================= 基础配置 =============================
API_KEY = "sk-16400cc458574e788397680e05f4b823"  
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

INPUT_CSV  = "data/species_list.csv"
OUTPUT_CSV = "data/species_with_description_fixed.csv"
CACHE_FILE = "data/epithet_cache.json"

MAX_WORKERS = 5 
MIN_INTERVAL = 1.0  
BATCH_SAVE = 20        

family_map = {
    "Crocodylidae": "crocodile",
    "Canidae": "dog",
    "Felidae": "cat",
    "Ursidae": "bear",
    "Cervidae": "deer",
    "Bovidae": "cow",
    "Equidae": "horse",
    "Hominidae": "ape",
    "Muridae": "rodent",
    "Sciuridae": "squirrel",
}

# ============================= initialization =============================
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}
print(f"🔹 已加载缓存：{len(cache)} 条记录")

last_call = 0.0  # 用于每线程限速控制
cache_updates = 0

def explain_epithet(epithet, retries=3, delay=3):
    """解释种加词含义，带限速与重试"""
    global last_call, cache_updates

    key = epithet.lower().strip()
    if key in cache:  
        return cache[key]

    prompt = f"""
    You are a biologist and Latin expert.
    Explain in English what the Latin or Greek epithet '{epithet}' means in a biological naming context.
    Give a very short phrase (3-10 words) describing its meaning such as
    'narrow-headed', 'from China', 'named after Anderson', 'black and white', etc.
    Output only the phrase, no extra commentary.
    """

    now = time.time()
    diff = now - last_call
    if diff < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - diff)

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            meaning = completion.choices[0].message.content.strip()
            cache[key] = meaning
            cache_updates += 1
            last_call = time.time()
            if cache_updates % BATCH_SAVE == 0:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            return meaning
        except Exception as e:
            time.sleep(delay)
            continue

    cache[key] = ""
    return ""

# ============================= Generate Description =============================
def generate_description(family, canonical_name):
    epithet = canonical_name.split()[-1]
    meaning = explain_epithet(epithet)
    fam = family_map.get(family, "animal")
    if not meaning:
        return f"a {fam}"
    m_clean = meaning.strip("'\" ")
    m_lower = m_clean.lower()

    if m_lower.startswith("named after"):
        # 使用原始 m_clean 回填，保留大小写
        return f"a {fam} {m_clean}"
    elif m_lower.startswith("from"):
        return f"a {fam} {m_clean}"
    elif any(m_lower.endswith(suf) for suf in ["ed", "like", "shaped"]):
        return f"a {m_clean} {fam}"
    else:
        return f"a {fam} that resembles {m_clean}"

# ============================= Main Process =============================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} records")

    df["canonicalName"] = df["canonicalName"].fillna("").astype(str)
    df["epithet"] = df["canonicalName"].str.split().str[-1].fillna("").astype(str)

    unique_epithets = df["epithet"].unique().tolist()
    print(f"Remaining {len(unique_epithets)} epithets need to explain")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(explain_epithet, e): e for e in unique_epithets}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Fetching epithets"):
            pass
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Cache updated with {len(cache)} entries")

    descriptions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating descriptions"):
        key = str(row["epithet"]).strip().lower()  
        meaning = cache.get(key, "")
        fam = row["family"]
        canon = row["canonicalName"]
        descriptions.append(generate_description(fam, canon))

    df["description"] = descriptions
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n Result saved to: {OUTPUT_CSV}")