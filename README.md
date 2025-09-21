Hashtagger: Keyword-Augmented Retrieval (KAG)
=============================================

Hashtagger is an experimental pipeline for Keyword-Augmented Retrieval (KAG) — 
a transparent alternative to RAG (Retrieval-Augmented Generation). Instead of 
opaque embeddings and vector DBs, Hashtagger uses human-readable hashtags to 
organize and retrieve documents.

The pipeline lives in a single script, tag-pipeline.py, which can:
- Parse a Markdown file of stories into separate directories
- Extract motifs, entities, and themes with spaCy + clustering
- Generate hashtags for each story
- Build a hashtag-based directory index with symlinks
- Provide an interactive chat mode backed by mlx-lm (optimized for Apple Silicon / Metal)

------------------------------------------------------------
Features
------------------------------------------------------------
- All-in-one script: run parsing, tagging, indexing, and chatting in one place.
- Transparent retrieval: hashtags are stored as directories you can inspect, edit, or extend manually.
- Flexible augmentation: you can add documents under hashtags even if they’re not auto-tagged.
- MLX-powered chat: runs locally on Mac Metal with models like microsoft/Phi-3-mini-4k-instruct.

------------------------------------------------------------
Pipeline Overview
------------------------------------------------------------
All functionality is inside tag-pipeline.py:

1. Parse Markdown
   Splits allpages.md into stories_out/StoryTitle/story.txt

2. Discover Motifs
   Uses spaCy + SentenceTransformers + clustering to extract recurring characters, phrases, and themes.

3. Tag Stories
   Assigns hashtags per story, saved to hashtags.json

4. Build Index
   Creates hashtags/#TagName/ directories with symlinks to stories.

5. Chat Mode
   Interactive loop: queries hashtags → retrieves context → passes it into mlx-lm → returns a response.

------------------------------------------------------------
Installation
------------------------------------------------------------
Clone the repo:

    git clone https://github.com/jahbini/hashtagger.git
    cd hashtagger

Set up a virtual environment and install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

------------------------------------------------------------
Usage
------------------------------------------------------------
Run the full pipeline:

    python tag-pipeline.py

This will:
- Parse allpages.md
- Extract motifs
- Tag stories with hashtags
- Build the hashtag index in ./hashtags/

Start chat mode:

    python tag-pipeline.py chat

Example session:

    You: tell a story like southwick
    📚 Retrieved 7 docs from: ['#Like', '#Southwick']
    🤖 LLM: (model response here...)

------------------------------------------------------------
Example Directory Layout
------------------------------------------------------------
```
stories_out/
  └── Wisdom_and_The_Path/
      ├── story.txt
      └── hashtags.json
hashtags/
  ├── #Wisdom/
  │   └── Wisdom_and_The_Path.txt -> ../../stories_out/Wisdom_and_The_Path/story.txt
  ├── #Enlightenment/
  │   └── Wisdom_and_The_Path.txt -> ../../stories_out/Wisdom_and_The_Path/story.txt
```
------------------------------------------------------------
Why KAG?
------------------------------------------------------------
Traditional RAG pipelines hide retrieval in embeddings + vector DBs. 
Hashtagger instead:
- Keeps retrieval human-readable
- Allows manual curation (advertisements, cross-links, easter eggs)
- Creates training data slices naturally (e.g. all #Wisdom stories)

This makes it a natural complement to LoRA fine-tuning:
- LoRA → voice + style
- KAG → facts + motifs

------------------------------------------------------------
Roadmap
------------------------------------------------------------
- Add fuzzy-matching to suggest nearest hashtags if no exact match is found
- Support multiple inference backends (MLX-LM, HuggingFace, OpenAI)
- Log chat sessions into training-ready JSONL for incremental fine-tuning

------------------------------------------------------------
License
------------------------------------------------------------
MIT License (or your preferred license)
