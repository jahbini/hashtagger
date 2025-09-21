# Keyword-Augmented Retrieval (KAG): Advantages Over Traditional RAG

## Abstract
Keyword-Augmented Retrieval (KAG) is a transparent, human-centric alternative to traditional Retrieval-Augmented Generation (RAG).  
Where RAG systems rely on opaque vector embeddings and dense similarity search, KAG organizes knowledge through human-readable hashtags.  
This approach maintains interpretability, supports manual curation, and enables hybrid systems that combine fine-tuned models with live, editable knowledge bases.

---

## Introduction
RAG pipelines have become a popular method to extend the knowledge of large language models (LLMs).  
However, they suffer from two key limitations:

1. **Opacity** — embeddings are not human-auditable, making it difficult to verify what concepts are represented.  
2. **Rigidity** — changes to documents require re-embedding, and users cannot easily inject context manually.

Keyword-Augmented Retrieval addresses both issues by grounding retrieval in keywords and hashtags that are meaningful to humans.  
This allows experts, curators, or even casual users to directly shape the system’s retrieval behavior.

---

## Advantages of KAG

### 1. Transparency
- Hashtags are **human-readable**.  
- Users can see exactly why a document was retrieved.  
- Debugging retrieval is straightforward compared to inspecting embedding vectors.

### 2. Flexibility
- A document can be tagged with any number of hashtags.  
- Users can add, remove, or reassign hashtags **without re-embedding**.  
- Cross-links and even advertisements can be introduced intentionally.

### 3. Human-Centered Semantics
- Hashtags capture **cultural meaning**, memes, and motifs in ways embeddings cannot.  
- Recurrent themes (e.g. *#Wisdom*, *#Southwick*) provide stable anchors for retrieval.  
- Keywords can come from **users as well as models**, mirroring practices on Twitter, YouTube, and other platforms.

### 4. Compatibility with LoRA / Fine-Tuning
- Fine-tuning adapts the model’s **style and voice**.  
- KAG supplies **facts, motifs, and evolving context**.  
- Together, they form a layered system:  
  - **LoRA → voice + reasoning style**  
  - **KAG → dynamic, curated content grounding**

### 5. Lightweight Infrastructure
- No need for vector databases or embedding refresh jobs.  
- Hashtag directories and symlinks can be maintained with standard file systems.  
- Works well on constrained environments (e.g. local Mac Metal with `mlx-lm`).

---

## Example Workflow
1. Parse a Markdown corpus into individual stories.  
2. Extract motifs and themes with NLP tools.  
3. Assign hashtags to each story, stored in `hashtags.json`.  
4. Build a hashtag directory index (`#Wisdom`, `#Enlightenment`, etc.).  
5. During chat, user queries are mapped to hashtags → documents are retrieved → context is passed to the LLM.  

This ensures retrieval remains explainable and adaptable without retraining embeddings.

---

## Conclusion
Keyword-Augmented Retrieval (KAG) offers a practical, human-auditable, and flexible retrieval mechanism that complements fine-tuned LLMs.  
By shifting focus from opaque vectors to transparent hashtags, KAG enables both **expert curation** and **grassroots participation** in shaping AI systems.  

KAG provides a viable path forward for systems that value interpretability, adaptability, and integration with evolving human knowledge.

---

## Citation
Hinds, J.A., et al. *Keyword-Augmented Retrieval (KAG): Transparent Hashtag-Based Knowledge Grounding for LLMs.*  
[GitHub Repository: jahbini/hashtagger](https://github.com/jahbini/hashtagger)
