# Text-to-SQL with a from-scratch Transformer (WikiSQL)

An English question plus the column names of a table go in; a SQL query over that table comes out.
The model is the original encoder–decoder Transformer of *Attention Is All You Need* (Vaswani et al., 2017),
written from basic PyTorch layers, trained once from random initialisation on
[WikiSQL](https://github.com/salesforce/WikiSQL), and scored with WikiSQL's official evaluator
by executing the generated queries on the real tables.

```
Question  what is terrence ross' nationality
Columns   Player, No., Nationality, Position, Years in Toronto, School/Club Team
Model     select <c2> where <c0> = terrence ross
SQL       SELECT "Nationality" FROM table WHERE "Player" = 'terrence ross'
```

**Medium blog:** _<link>_  ·  **LinkedIn post:** _<link>_

---

## Results

All numbers are produced by the scripts and collected in [`results/tables.md`](results/tables.md)
(generated from [`results/metrics.json`](results/metrics.json)). Paste the final tables here after the run.

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | | | |
| Dev | beam (4) | | | |
| Test | _chosen on dev_ | | | |

Reference points from the assignment: the LSTM seq2seq baseline of the WikiSQL paper reaches ≈36% test
execution accuracy; models with pretrained encoders exceed 90%.

Figures (in `results/figures/`): positional encoding · loss curves · learning-rate schedule ·
cross-attention map · front end. Qualitative samples: [`results/samples.md`](results/samples.md).

---

## Repository layout

```
starter/                 the four given files + check_starter.py, byte-identical (verified by scripts/starter.sha256)
model/attention.py       ScaledDotProductAttention, MultiHeadAttention
model/layers.py          PositionwiseFeedForward, EncoderLayer, DecoderLayer
model/transformer.py     masks (padding, causal), Encoder, Decoder, Transformer (3-way weight sharing)
train.py                 Task 3: teacher forcing, label smoothing, Adam + Noam schedule, best-dev checkpoint
decode.py                Task 4: greedy, beam search (k=4), parser -> WikiSQL dict, readable SQL
evaluate_model.py        Task 5: prediction files, official evaluator, component accuracy, attention map, samples, tables
data_report.py           Task 1.3–1.4: Table 1, starter shapes, positional-encoding heat-map
checks.py                Section 3.2 correctness checks -> results/correctness_checks.md
common.py                paths, fixed hyper-parameters, shared helpers
app/app.py               Task 6.1: Gradio front end
results/                 prediction files, metrics.json, tables.md, figures/, samples.md, train_log.csv
notebooks/               one-click GPU run for Kaggle / Colab
scripts/                 data setup, full pipeline, compliance check
docs/                    viva notes, blog and LinkedIn drafts
```

---

## How to run

Tested with Python 3.11/3.12 and PyTorch 2.x.

```bash
pip install -r requirements.txt
bash scripts/setup_data.sh                                    # clone WikiSQL, extract, link into starter/

# Task 1
cd starter && python data_prep.py && python tokenizer.py && python check_starter.py && cd ..
python data_report.py                                         # Table 1 + positional-encoding heat-map

python checks.py                                              # correctness checks (no training needed)
python train.py                                               # Task 3: 20 epochs on GPU (~25–40 min on a T4)
python checks.py                                              # same checks on the trained checkpoint
python evaluate_model.py dev                                  # Tasks 4–5 on dev: greedy + beam, everything else
python evaluate_model.py test                                 # test, exactly once, decoding chosen on dev
python app/app.py                                             # front end at http://127.0.0.1:7860
```

`bash scripts/run_all.sh` runs the whole pipeline; `notebooks/train_on_gpu.ipynb` does the same on Kaggle/Colab.
The official evaluator can also be called by hand exactly as in the assignment:

```bash
cd WikiSQL && python evaluate.py data/dev.jsonl data/dev.db ../results/dev_greedy.jsonl
```

---

## Implementation

**Model (Task 2).** d_model 256, 4 heads (d_k = d_v = 64), 3 + 3 layers, d_ff 1024, dropout 0.1, post-norm
`LayerNorm(x + Dropout(Sublayer(x)))`. One `TokenEmbedding` instance feeds the encoder `InputLayer`, the decoder
`InputLayer` and, through `self.generator.weight = self.shared.emb.weight`, the output projection.
**7,577,600 trainable parameters** (embedding 2,048,000 · encoder 2,369,280 · decoder 3,160,320; no output bias).
Masks are boolean (True = may attend). Masked scores are filled with the most negative finite value of the
score dtype, which is −∞ after softmax (weights are exactly 0) but cannot create NaNs and survives fp16.
Embeddings are initialised N(0, d_model^-0.5) so that after the starter's √d_model scaling they have unit
variance, like the positional encodings; all other matrices use Xavier-uniform.

**Training (Task 3).** Teacher forcing (`tgt[:, :-1]` → `tgt[:, 1:]`), cross-entropy with ε_ls = 0.1 ignoring
`<pad>`, Adam (0.9, 0.98, 1e-9), the paper's schedule with 4000 warm-up steps stepped every batch, batch 64,
20 epochs. Train loss, dev loss and learning rate are logged after every epoch (`results/train_log.csv`); only the
lowest-dev-loss checkpoint is kept. fp16 autocast is used on GPU for speed (the loss is computed in fp32).
The test set is not loaded by `train.py`.

**Decoding (Task 4).** Batched greedy decoding and beam search with beam 4 and the paper's length penalty
(α = 0.6, Sec. 6.1); both stop at `</s>` or 64 tokens. Two decoding-time rules, using only the model input:

- *Validity mask.* `<pad>`, `<s>`, `<sep>` and column tokens `<cK>` with K ≥ number of table columns get
  −∞ before the arg-max / top-k. A query that names a non-existent column cannot be executed, and the column
  count is known at inference time (the front end gets it from the user's column list).
- *`<unk>` repair.* Characters never seen in training (Hebrew, Arabic, rare CJK…) become `<unk>`. Since WHERE
  values are copied from the question, the parser rebuilds such a value by matching the known pieces around
  each `<unk>` against the question. This raises the gold round-trip from 99.49% to 99.96% execution accuracy.

The parser turns the decoded string into `{"sel", "agg", "conds"}` (values may contain the word "and"; splits
happen only on `and <cK>`), writes exactly one line per example in file order, and writes `{"error": "parse"}`
when the string is not a valid query. `to_readable_sql` restores the real column names.

**Evaluation (Task 5).** `evaluate_model.py dev` decodes dev with both strategies, runs the unmodified official
evaluator, computes component accuracy (sel / agg / WHERE as a set), and picks the strategy with the higher dev
execution accuracy as the final decoding. `evaluate_model.py test` refuses to run before that choice exists and
refuses to run a second time, so the test set is used once.

---

## Correctness checks

Output of `python checks.py` (regenerate after training and paste `results/correctness_checks.md` here):

- **Causal mask:** PASS. Changing the last decoder input token: max |Δ| at earlier positions = 0.00e+00.
- **Padding mask:** PASS. Appending 25 `<pad>` tokens to every source: max |Δ| of the output logits = 0.00e+00.
- **Attention rows sum to 1:** PASS. Max |row sum over unmasked keys − 1| = 2.4e-07; max weight on a masked key = 0.
- **Weight sharing:** PASS. `model.generator.weight is model.shared.emb.weight` → True.
- **Learning rate:** PASS. Peak 9.882e-04 at step 4000; exactly linear before, exactly ∝ step^-0.5 after.
- **Gold round-trip:** PASS. Official evaluator on parsed gold dev targets: execution 99.96%, logical form 99.96%,
  parse failures 0.00%.

---

## Front end

`python app/app.py` (add `--share` on Colab/Kaggle). Type a question and a comma-separated column list; the app
calls the trained checkpoint and shows the SQL with real column names, the raw model output and the
column-token mapping.

![front end](results/figures/frontend.png)

---

## Constraints

- No `nn.Transformer*`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging Face `transformers`
  or pretrained weights anywhere: `bash scripts/check_compliance.sh` scans the code and verifies that `starter/`
  is byte-identical to the given files.
- Dev is used only for checkpoint selection and the greedy-vs-beam choice; test is decoded once.

## Team

| Member | Main ownership |
|---|---|
| _name_ | model/, checks.py, attention analysis |
| _name_ | train.py, decode.py, evaluation, front end |

## References

Vaswani et al. (2017), *Attention Is All You Need*. Zhong, Xiong and Socher (2017), *Seq2SQL* / WikiSQL.
