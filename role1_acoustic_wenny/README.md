# Role 1: Acoustic Similarity — Wenny

## Goal

Build a second audio embedding space that captures **low-level acoustic features** (timbre, rhythm, texture) rather than CLAP's high-level semantic features. This enables **audio-to-audio retrieval** — "find songs that sound like this one."

## Background

CLAP maps audio into a space shared with text, so it captures semantic meaning (genre, mood, vibe). OpenL3, by contrast, learns purely acoustic representations — two songs that sound similar will be close together even if they're in different genres. Your role produces a complementary view of the same dataset.

## Your Tasks

1. **Implement `src/embeddings/openl3.py`**
   - Subclass `EmbeddingGenerator` from `src/embeddings/base.py`
   - Use OpenL3 (`content_type="music"`, `embedding_size=512`)
   - Process all 8,000 tracks in the FMA small subset
   - Save embeddings to `data/processed/openl3_embeddings.npy`
   - Save track IDs to `data/processed/openl3_track_ids.npy`

2. **Build a FAISS index**
   - Use `src/indexing/faiss_index.py` to build an `IndexFlatIP` index
   - Save to `data/processed/openl3_faiss.index`

3. **Test audio-to-audio queries**
   - Pick 5–10 seed tracks from different genres
   - Find their top-10 nearest neighbours
   - Report whether the results are musically coherent (check genre distribution of results)

4. **t-SNE comparison notebook**
   - Plot OpenL3 t-SNE coloured by genre (same style as `notebooks/03_embedding_visualisation.ipynb`)
   - Place side-by-side with the CLAP t-SNE
   - Do the genre clusters agree? Where do they differ?
   - Key question: CLAP places Hip-Hop and Pop at 0.81 cosine similarity — does OpenL3 separate them better?

5. **Genre centroid similarity heatmap**
   - Compute cosine similarity between the 8 genre centroids (same genres as CLAP analysis)
   - Compare the heatmap to CLAP's — which genres are more/less separable acoustically?

## Setup

```bash
# Install OpenL3
pip install openl3

# Or if using MusicFM instead:
# pip install musicfm  (check GitHub for install instructions)
```

## Key Files

| Path | Purpose |
|---|---|
| `src/embeddings/base.py` | Base class to subclass |
| `src/embeddings/clap.py` | Reference implementation |
| `src/indexing/faiss_index.py` | FAISS index wrapper |
| `src/config.py` | Paths and constants |
| `src/audio_utils.py` | Resolve MP3 paths from track IDs |
| `data/fma_metadata/tracks.csv` | Track metadata |
| `data/fma_small/` | Audio files |
| `data/processed/clap_embeddings.npy` | CLAP embeddings (for comparison) |

## Deliverables

- [ ] `src/embeddings/openl3.py`
- [ ] `data/processed/openl3_embeddings.npy`
- [ ] `data/processed/openl3_faiss.index`
- [ ] Notebook: t-SNE comparison (CLAP vs OpenL3)
- [ ] Notebook: audio-to-audio retrieval demo
- [ ] Genre centroid heatmap (saved to `data/processed/`)
