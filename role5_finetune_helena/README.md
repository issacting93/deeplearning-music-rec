# Role 5: Fine-tuning & Deep Analysis — Helena

## Goal

Improve the base CLAP model by **fine-tuning it on FMA data**, and conduct deeper analysis of the embedding space to understand what the model has and hasn't learned. This role feeds an improved embedding back into the whole pipeline.

## Background

The existing CLAP embeddings (512-dim, 7,997 tracks) were generated using the pretrained `music_audioset_epoch_15_esc_90.14.pt` checkpoint with no FMA-specific adaptation. Fine-tuning on FMA with contrastive learning — pairing each audio clip with a text description derived from its metadata — should improve retrieval quality, especially for genres that are currently conflated (e.g. Hip-Hop and Pop at 0.81 cosine similarity).

## Your Tasks

### Part A: Fine-tuning CLAP on FMA

1. **Construct training pairs**
   - For each track, build a text description:
     ```
     "{title} by {artist}. {genre} music."
     ```
   - Use tags and track information fields if available for richer descriptions
   - Split 8,000 tracks into train (80%) / val (10%) / test (10%) — save the split to `data/processed/splits.json` so all roles use the same split

2. **Fine-tune the CLAP model**
   - Load the pretrained checkpoint using `laion_clap.CLAP_Module`
   - Use **InfoNCE / NTXentLoss** (contrastive loss): audio and its paired text should be pulled together, while other pairs in the batch are pushed apart
   - Recommended batch size: 32–64 (larger = more negatives = better contrastive learning)
   - Fine-tune for 5–15 epochs; use early stopping on validation loss
   - Save checkpoints to `models/clap_finetuned_epoch{N}.pt`

3. **Generate fine-tuned embeddings**
   - After fine-tuning, regenerate embeddings for all 8,000 tracks
   - Save to `data/processed/clap_finetuned_embeddings.npy`
   - Build a new FAISS index: `data/processed/clap_finetuned_faiss.index`

### Part B: Before/After Comparison

4. **Visualisation comparison**
   - Regenerate t-SNE and PCA plots with the fine-tuned embeddings
   - Place side-by-side with the original CLAP t-SNE (`data/processed/tsne_clap_embeddings.png`)
   - Regenerate the genre centroid similarity heatmap — did the Hip-Hop/Pop similarity (0.81) decrease?
   - Save all updated plots to `data/processed/` with `_finetuned` suffix

5. **Retrieval quality comparison**
   - Re-run the same 8 test queries from the main README using fine-tuned embeddings
   - Compare top-5 results: do they improve?
   - Report P@10 and NDCG@10 before/after (coordinate with Jiayi's evaluation framework)

### Part C: Deep Analysis

6. **Echo Nest feature correlation**
   - Load `echonest.csv` (danceability, energy, valence, tempo, speechiness, acousticness)
   - Merge with small subset tracks — get the overlapping ~5k tracks
   - Run PCA on CLAP embeddings, get PC scores for each track
   - Compute Pearson correlation between each PC and each Echo Nest feature
   - Which PCA dimensions encode danceability? Energy? Valence?
   - Plot as a correlation heatmap: `data/processed/pca_echonest_correlation.png`

7. **Failure analysis**
   - Identify systematic retrieval failures using the original CLAP index
   - For each of the 8 test queries, look at the top-20 results and count genre mismatches
   - Find tracks that are **outliers** in embedding space (furthest from their genre centroid)
   - What makes these tracks unusual? (check their metadata, listen if possible)

8. **Ablation study** (if time permits)
   - Compare HTSAT-tiny vs HTSAT-base backbone (different `amodel` argument in `CLAP_Module`)
   - Try embedding dimension 512 vs 256
   - Report retrieval quality at each setting

## Setup

```bash
# CLAP is already installed; for fine-tuning you may need:
pip install torchvision  # required by laion-clap
```

## Key Files

| Path | Purpose |
|---|---|
| `src/embeddings/clap.py` | Existing CLAP embedding pipeline |
| `src/metadata.py` | Load and filter tracks |
| `src/config.py` | Paths and constants |
| `data/fma_metadata/tracks.csv` | Track metadata for text descriptions |
| `data/fma_metadata/echonest.csv` | Echo Nest audio features |
| `data/processed/clap_embeddings.npy` | Original embeddings (baseline) |
| `data/processed/pca_clap_embeddings.png` | Original PCA plot (for comparison) |
| `data/processed/tsne_clap_embeddings.png` | Original t-SNE plot (for comparison) |
| `data/processed/genre_similarity_heatmap.png` | Original heatmap (for comparison) |

## Deliverables

- [ ] `src/finetune/train_clap.py` — fine-tuning script
- [ ] `data/processed/splits.json` — shared train/val/test split
- [ ] `models/clap_finetuned_epoch{N}.pt` — fine-tuned checkpoint
- [ ] `data/processed/clap_finetuned_embeddings.npy`
- [ ] `data/processed/clap_finetuned_faiss.index`
- [ ] Notebook: before/after comparison (t-SNE, heatmap, retrieval results)
- [ ] Notebook: Echo Nest feature correlation analysis
- [ ] Notebook: failure analysis and outlier tracks
