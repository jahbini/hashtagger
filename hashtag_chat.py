import os
import re
import json
import spacy
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import subprocess

# --- Setup ---
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~22MB, fast and good enough

HOUSE_MOTIFS = [
    "Wisdom", "Enlightenment", "Celarien",
    "Logos", "Ethos", "Pathos", "Anima",
    "Tarot", "Energy"
]

GENERIC_BLOCKLIST = {"low", "people", "individuals", "part", "thing", "stuff"}

# -------- PASS 0: Parse Markdown into stories --------
def parse_markdown(md_path, outdir="stories_out"):
    Path(outdir).mkdir(exist_ok=True)
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    stories = {}
    current_title, buffer = None, []

    for line in lines:
        if line.startswith("# "):  # new story
            if current_title:
                stories[current_title] = "\n".join(buffer).strip()
                save_story(outdir, current_title, buffer)
                buffer = []
            current_title = line[2:].strip()
        else:
            buffer.append(line.strip())

    if current_title and buffer:
        stories[current_title] = "\n".join(buffer).strip()
        save_story(outdir, current_title, buffer)

    return stories

def safe_dirname(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:50]

def save_story(outdir, title, content_lines):
    dirname = Path(outdir) / safe_dirname(title)
    dirname.mkdir(parents=True, exist_ok=True)
    with open(dirname / "story.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))

# -------- Embeddings --------
def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

# -------- PASS 1: Discover Motifs --------
def discover_motifs(stories, k_clusters=5):
    all_entities, all_phrases, story_texts = [], [], []
    titles = list(stories.keys())

    for title, text in stories.items():
        doc = nlp(text)
        story_texts.append(text)

        ents = [ent.text for ent in doc.ents if ent.label_ in ["PERSON","ORG","GPE","WORK_OF_ART"]]
        all_entities.extend(ents)

        chunks = [chunk.text.strip() for chunk in doc.noun_chunks if 1 <= len(chunk.text.split()) <= 4]
        all_phrases.extend(chunks)

    top_entities = [e for e, c in Counter(all_entities).items() if c > 2]
    top_phrases  = [p for p, c in Counter(all_phrases).items() if c > 3]

    embeddings = [get_embedding(txt) for txt in story_texts]
    kmeans = KMeans(n_clusters=k_clusters, random_state=42).fit(embeddings)
    cluster_ids = kmeans.labels_

    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(story_texts)
    terms = tfidf.get_feature_names_out()
    cluster_labels = []
    for i in range(k_clusters):
        idx = np.where(cluster_ids == i)[0]
        if len(idx) == 0:
            continue
        centroid = X[idx].mean(axis=0)
        top_idx = np.array(centroid.A).ravel().argsort()[-3:]
        cluster_labels.append([terms[j] for j in top_idx])

    motifs = {
        "house_motifs": HOUSE_MOTIFS,
        "characters": top_entities,
        "phrases": top_phrases,
        "themes": cluster_labels
    }

    with open("motifs.json", "w", encoding="utf-8") as f:
        json.dump(motifs, f, indent=2)

    return motifs, kmeans, embeddings, titles

# -------- PASS 2: Clean + Tag Stories --------
def clean_tag(tag: str) -> str | None:
    t = tag.lstrip("#").strip()

    if not t or t.startswith(("/", "*")) or t.startswith(("##", "###")):
        return None
    if len(t) < 3 or len(t) > 40:
        if t.upper() not in ["AI", "LLM"]:
            return None

    doc = nlp(t)
    if len(doc) == 1:
        t = doc[0].lemma_

    if t.lower() in [m.lower() for m in HOUSE_MOTIFS]:
        return "#" + t.capitalize()
    if t.lower() in STOP_WORDS or t.lower() in GENERIC_BLOCKLIST:
        return None

    if t.isupper():
        clean = t
    else:
        clean = t.capitalize()

    return "#" + clean

def tag_story(title, text, motifs, kmeans, embeddings, titles):
    doc = nlp(text)
    tags = set()

    for ent in doc.ents:
        if ent.label_ in ["PERSON","ORG","GPE","WORK_OF_ART"]:
            tags.add("#" + ent.text.replace(" ", ""))

    for p in motifs["phrases"] + motifs["characters"]:
        if p.lower() in text.lower():
            tags.add("#" + p.replace(" ", ""))

    idx = titles.index(title)
    cluster_id = kmeans.predict([embeddings[idx]])[0]
    cluster_tags = motifs["themes"][cluster_id]
    for t in cluster_tags:
        tags.add("#" + t.replace(" ", ""))

    for motif in motifs.get("house_motifs", []):
        if motif.lower() in text.lower():
            tags.add("#" + motif)

    cleaned = set()
    for tag in tags:
        c = clean_tag(tag)
        if c:
            cleaned.add(c)

    return {"title": title, "hashtags": sorted(cleaned)}

def save_story_tags(outdir, title, tags):
    dirname = Path(outdir) / safe_dirname(title)
    with open(dirname / "hashtags.json", "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2)

# -------- PASS 3: Build Hashtag Index --------
def build_hashtag_index(stories_dir="stories_out", hashtags_dir="hashtags"):
    base = Path(stories_dir)
    tagbase = Path(hashtags_dir)
    tagbase.mkdir(exist_ok=True)

    for story_dir in base.iterdir():
        if not story_dir.is_dir():
            continue
        tags_file = story_dir / "hashtags.json"
        story_file = story_dir / "story.txt"
        if not tags_file.exists() or not story_file.exists():
            continue

        with open(tags_file, "r", encoding="utf-8") as f:
            tags = json.load(f)["hashtags"]

        for tag in tags:
            tagdir = tagbase / tag
            tagdir.mkdir(parents=True, exist_ok=True)
            linkpath = tagdir / f"{story_dir.name}.txt"
            if not linkpath.exists():
                try:
                    os.symlink(story_file.resolve(), linkpath)
                except FileExistsError:
                    pass

# -------- PASS 4: Chat with MLX-LM --------
def run_llm(prompt, model="microsoft/Phi-3-mini-4k-instruct", max_tokens=300):
    """
    Call mlx-lm locally (Metal/MLX backend).
    Ensure you've installed mlx-lm and downloaded the model:
      pip install mlx-lm
      python -m mlx_lm.download microsoft/Phi-3-mini-4k-instruct
    """
    print("JIM",prompt)
    try:
        result = subprocess.run(
            [
                "python", "-m", "mlx_lm","generate",
                "--model", model,
                "--prompt", prompt,
                "--max-tokens", str(max_tokens)
            ],
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        return f"[MLX error: {e}]"

def chat(hashtags_dir="hashtags", model="microsoft/Phi-3-mini-4k-instruct"):
    print("Keyword-augmented chat (MLX-LM). Type 'quit' to exit.")
    while True:
        q = input("\nYou: ")
        if q.strip().lower() in ("quit", "exit"):
            break
        words = re.findall(r"\w+", q)
        hits = []
        for w in words:
            tagdir = Path(hashtags_dir) / ("#" + w.capitalize())
            if tagdir.exists():
                hits.append(tagdir)

        if not hits:
            print("🤔 No matching hashtags found.")
            continue

        seen, context = set(), []
        for tagdir in hits:
            for link in tagdir.glob("*.txt"):
                if link.resolve() in seen:
                    continue
                seen.add(link.resolve())
                with open(link.resolve(), "r", encoding="utf-8") as f:
                    context.append(f.read())

        print(f"\n📚 Retrieved {len(context)} docs from: {[t.name for t in hits]}")

        joined_context = "\n\n".join(context)[:3000]
        prompt = f"""You are a helpful assistant.
Here is some background context from my files:
{joined_context}

Now, answer the question:
{q}
"""
        answer = run_llm(prompt, model=model)
        print(f"\n🤖 LLM: {answer}\n")

# -------- MAIN --------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat()
    else:
        stories = parse_markdown("allpages.md", outdir="stories_out")
        motifs, kmeans, embeddings, titles = discover_motifs(stories, k_clusters=5)

        for title, text in stories.items():
            tagged = tag_story(title, text, motifs, kmeans, embeddings, titles)
            save_story_tags("stories_out", title, tagged)
            print(f"{title} => {', '.join(tagged['hashtags'])}")

        build_hashtag_index("stories_out", "hashtags")
        print("✅ Hashtag index built in ./hashtags/")
