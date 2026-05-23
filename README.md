# Tamil Sign Language Fingerspelling Recognition: CNN vs Vision Transformer

A comparative study of **EfficientNet-B3** and **Swin-Small** on the **TLFS23** dataset (247 Tamil fingerspelling letter classes + 1 background class), produced for **ST456 Deep Learning** at the LSE.

The central question: do architectural differences between a CNN and a Vision Transformer produce *systematic*, *measurable* gaps in error structure across character complexity, or only in headline accuracy?

---

## TL;DR

Results on a held-out test split of a 200-images-per-class subset of TLFS23 (~49,600 images, 248 classes). Numbers are final pending project review.

| Model | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | σ(per-class F1) |
|---|---|---|---|---|---|
| EfficientNet-B3 | 0.5177 | 0.8738 | 0.4666 | 0.4667 | 0.24 |
| Swin-Small | **0.9935** | **0.9993** | **0.9935** | **0.9935** | **0.01** |

McNemar's test on 7,417 paired test predictions: **χ²(1) = 3497.27, p < 0.001** (3,544 images correct under Swin but not EfficientNet-B3, vs. 15 in the reverse direction).

The 47.6-point Top-1 gap is statistically significant, but the more interesting result is the **dissociation in error structure**. EfficientNet-B3 shows a monotonic per-tier F1 gradient (Simple > Moderate > Complex), consistent with prior work on smaller alphabets; Swin's per-tier profile is essentially flat. Swin dominates 245 of 247 classes and resolves 19 of EfficientNet-B3's 20 most-confused class pairs — pairs that share a base consonant in the Uyir-Mei compositional system. Grad-CAM on EfficientNet-B3 misclassifications shows attention drifting onto task-irrelevant background (chair edges, shirt prints) rather than within-hand modifier strokes — the locality-prior failure mode predicted when the discriminating signal is global and compositional. The Swin number is high enough that we flag a potential validation-leakage confound — see the report.

To our knowledge, this is the first published deep-learning benchmark on the **full 247-class TLFS23 task**; the only prior work using TLFS23 (Sangeetha & Divya Gowri, 2025) collapsed it to 13 vowel classes.

---

## Repository contents

```
.
├── CNN_final.ipynb          # EfficientNet-B3 pipeline: preprocessing, staged fine-tuning, evaluation
├── Transformer_final.ipynb  # Swin-Small pipeline: preprocessing, training, evaluation
├── report.pdf               # Final 8-page report (ICML two-column format)
└── README.md
```

**Not included:** the TLFS23 dataset, trained weights, or cached predictions. The dataset is available on Mendeley (see citation below); weights and predictions can be regenerated from the notebooks.

---

## Methodology in brief

- **Dataset.** TLFS23 (Mendeley), 248 classes (247 letters + Background). We work with a 200 images/class subset (~49,600 images) for Colab training-time feasibility, with stratified train/val/test splits.
- **Complexity partition.** Classes are grouped into Simple / Moderate / Complex tiers via empirical inter-class cosine similarity in pixel space — a methodological choice that lets the partition emerge from the data rather than from linguist-assigned categories.
- **Two preprocessing pipelines.** EfficientNet uses `[-1, 1]` normalisation via `preprocess_input`; Swin uses `[0, 1]`. These are kept strictly separate to preserve fair comparison.
- **Training.** TensorFlow / Keras, mixed precision (`mixed_float16`), A100 GPU on Colab. EfficientNet uses staged fine-tuning (head-only → partial unfreezing → full); Swin trains end-to-end from pretrained ImageNet weights.
- **Evaluation.** Top-1 and Top-5 accuracy, per-class F1, confusion matrix analysis, McNemar's test on paired predictions, and Grad-CAM visualisations. The Grad-CAM sample is illustrative rather than evidential — quantification is flagged as outstanding work in the report.

---

## Reproduction

1. **Get the data.** Download TLFS23 from Mendeley (link in the report references) and place it under a Drive directory accessible to your Colab runtime. Both notebooks expect a Shared Drive layout at `/content/drive/Shareddrives/...`; adjust the path constants at the top of each notebook to match your setup.
2. **Run `CNN_final.ipynb` end to end** for the EfficientNet-B3 results. Training takes roughly a few hours on an A100.
3. **Run `Transformer_final.ipynb` end to end** for the Swin-Small results. Same compute requirements.
4. **Read `report.pdf`** for the full analysis, including the cross-architecture comparison, the prior-work table (§5.5), and the discussion of confounds.

A note on a non-obvious bug we hit: `label_to_idx.json` was emitted with 1-based indices while the model expected 0-based, causing training to collapse to random chance until corrected. The notebooks now apply a `LABEL_OFFSET` constant consistently — keep this if you reuse the label map elsewhere.

---

## Falsifiable predictions

The report commits to two predictions that this work invites future work to test:

1. The per-tier F1 gradient observed for EfficientNet should reappear in any CNN architecture trained under comparable budgets on the full 247-class task — i.e., the gradient is a property of convolutional inductive bias plus task complexity, not of EfficientNet specifically.
2. The flat per-tier profile for Swin should narrow but not invert under a stricter validation protocol that rules out the leakage confound.

---

Code released under the MIT License. The TLFS23 dataset is governed by the licence on its Mendeley listing — see there for terms.
