# Role 2: Lyrics & Semantic Search — Sid

## Goal

Build a **text-based retrieval view** using track metadata and, optionally, lyrics. This enables search like "find songs about loss and moving on" or "songs by artists in the folk tradition" — things that audio alone cannot capture.

## Background

FMA does not include lyrics, but `tracks.csv` contains rich text fields: title, artist, genre, tags, and information/description fields. These can be embedded with Sentence-BERT to create a semantic search index. Optionally, lyrics can be fetched from the Genius API for a subset of tracks to create a richer signal.

## Your Tasks

1. **Build metadata text strings**
   - For each of the 8,000 small-subset tracks, construct a text string:
     ```
     "{title} by {artist}. Genre: {genre}. Tags: {tags}."
     ```
   - Pull fields from `tracks.csv` using `src/metadata.py`
   - Handle missing fields gracefully (many tracks have empty tags)

2. **Implement `src/embeddings/sbert.py`**
   - Subclass `EmbeddingGenerator` from `src/embeddings/base.py`
   - Use `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
   - Embed all 8,000 text strings in batches
   - Save to `data/processed/sbert_embeddings.npy` + `sbert_track_ids.npy`

3. **Build a FAISS index**
   - Save to `data/processed/sbert_faiss.index`
   - Test with text queries like "dreamy electronic instrumental", "acoustic folk singer-songwriter"

4. **Overlap analysis with CLAP**
   - For 10 test queries, run both CLAP (audio-based) and SBERT (text-based) retrieval
   - Compute rank correlation (Spearman's ρ) between the two result lists
   - Do they agree on the same tracks? Where do they diverge?

5. **Echo Nest feature exploration** (optional but encouraged)
   - `echonest.csv` has danceability, energy, valence, tempo, speechiness, acousticness for ~13k tracks
   - Merge with the small subset — how many overlap?
   - Can these numeric features act as additional retrieval filters? (e.g. "high energy, low valence")

6. **(Optional) Lyrics via Genius API**
   - Register for a free Genius API token at https://genius.com/api-clients
   - Match FMA tracks by artist + title (fuzzy matching recommended)
   - For tracks with lyrics found, embed lyrics separately and compare to metadata-only embeddings

## Setup

```bash
# Sentence-BERT is already in requirements.txt
# If fetching lyrics:
pip install lyricsgenius
```

## Key Files

| Path | Purpose |
|---|---|
| `src/embeddings/base.py` | Base class to subclass |
| `src/metadata.py` | Load tracks.csv |
| `src/indexing/faiss_index.py` | FAISS index wrapper |
| `src/config.py` | Paths and constants |
| `data/fma_metadata/tracks.csv` | Track metadata (title, artist, genre, tags) |
| `data/fma_metadata/echonest.csv` | Echo Nest audio features |

## Deliverables

- [ ] `src/embeddings/sbert.py`
- [ ] `data/processed/sbert_embeddings.npy`
- [ ] `data/processed/sbert_faiss.index`
- [ ] Notebook: semantic search demo (text queries → track results)
- [ ] Notebook: CLAP vs SBERT overlap analysis
- [ ] Optional: lyrics-enriched embeddings + comparison
