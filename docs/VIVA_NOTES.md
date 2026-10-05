# Viva notes

All marks come from the viva. Both members should be able to answer every question below from the code.

## Data and framing
- **Why column tokens?** The model points at `<cK>` instead of spelling a column name, so the output vocabulary
  for columns is 64 symbols and works on unseen tables (dev/test share no tables with train).
- **Source / target.** `question <sep> <c0> name <c1> name …` → `select [agg] <cK> [where <cK> op value (and …)*]`.
  Source gets `</s>`; target gets `<s> … </s>`.
- **Shared BPE (8,000).** One vocabulary for both sides makes three-way weight sharing possible and lets the
  decoder emit the same pieces it reads in the question when it copies a WHERE value.
- **19 train pairs dropped** (source > 160 or target > 64 tokens). Dev/test keep every row so predictions align
  line-by-line with the gold files.

## Attention (`model/attention.py`)
- `softmax(QKᵀ/√d_k)V`. Dividing by √d_k keeps the dot-product variance ≈ 1 so softmax does not saturate.
- Masked scores → `finfo(dtype).min`: exp underflows to exactly 0 (same as −∞) but a fully masked row would give a
  uniform distribution instead of NaN, and the value is valid in fp16.
- Multi-head: one `Linear(256, 256)` per Q/K/V equals 4 stacked `256×64` matrices; reshape to (B, h, L, 64),
  attend per head, concatenate, project with W_O.

## Masks (`model/transformer.py`)
- Source padding mask (B,1,1,S): used by encoder self-attention **and** decoder cross-attention.
- Target mask (B,1,T,T) = padding AND lower-triangular causal mask: decoder position t sees ≤ t.
- Query rows at padded positions still have valid keys, and their outputs are ignored by the loss.

## Layers
- Post-norm: `LayerNorm(x + Dropout(Sublayer(x)))`, as in the paper (no final LayerNorm in post-norm).
- FFN 256 → 1024 → ReLU → 256, applied per position.
- Dropout sites (paper Sec. 5.4): every sub-layer output before the residual add, and embeddings + positions
  (inside the starter `PositionalEncoding`).

## Weight sharing and parameter count
- `generator.weight = shared.emb.weight`: same `Parameter`, so `is` holds and gradients from both uses add up.
- 7,577,600 = 8000·256 (embedding) + 3·789,760 (encoder layer: 4·(256²+256) = 263,168 attention
  + FFN 525,568 + 2 LayerNorms 1,024) + 3·1,053,440 (decoder layer: two attentions + FFN + 3 LayerNorms).
  `model.parameters()` de-duplicates the tied tensor.

## Training
- Teacher forcing: input `tgt[:, :-1]`, labels `tgt[:, 1:]`.
- Label smoothing 0.1: target distribution 0.9 on the gold token + 0.1 spread over the vocabulary; hurts
  perplexity, helps generalisation/beam search. Our dev loss is label-smoothed, so it never reaches 0.
- Schedule: lr = 256^-0.5 · min(step^-0.5, step · 4000^-1.5); linear warm-up for 4000 steps (peak 9.88e-4),
  then ∝ 1/√step. Warm-up matters for post-norm Transformers: early Adam updates with a large LR diverge.
  `LambdaLR` counts completed steps from 0, hence `step + 1`.
- ≈ 881 steps/epoch × 20 = 17,620 steps. Checkpoint = lowest dev loss; test never loaded.

## Decoding
- Greedy: arg-max each step, batched, stop on `</s>` / 64 tokens.
- Beam (k = 4): keep the 4 best partial sequences by summed log-prob; finished hypotheses are ranked by
  log-prob / ((5 + len)/6)^0.6 (GNMT length penalty, used in the paper) so short outputs are not unfairly favoured.
- Validity mask: forbid `<pad>`, `<s>`, `<sep>`, and `<cK>` with K ≥ #columns. Uses only the input, no gold.
- `<unk>` repair: unseen characters decode to "⁇"; the value is recovered by matching the surrounding pieces in
  the question.

## Evaluation
- **Logical form**: same sel, agg and *set* of conditions (values compared lower-cased).
- **Execution**: run predicted and gold queries on `dev.db`; same result. Higher than LF because different
  queries can return the same answer (e.g. a redundant condition, or COUNT on a different column).
- Warning thresholds: EX < 20% → mask bug; parse failures > 5% → decoding bug; EX > 95% → leakage.
- Gold round-trip 99.96%: proves the parser + tokenizer pipeline loses almost nothing, so remaining errors are
  the model's.

## Likely failure modes to discuss
Wrong value span (the model has no copy mechanism; it must regenerate the value piece by piece), wrong column
when two columns have similar names, missing condition in multi-condition questions, COUNT vs no aggregation,
numeric values formatted differently from the table.
