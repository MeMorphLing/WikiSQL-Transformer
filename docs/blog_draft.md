# Teaching a Transformer to write SQL, from scratch

_Draft for Medium (target 800–1500 words). Replace every ⟨…⟩ with numbers from `results/tables.md`.
Figures to embed: `cross_attention.png` and `loss_curves.png` (plus optionally `positional_encoding.png`)._

## The problem
Most data lives in tables; most people can't write SQL. Given a question and a table's column names, can a small
model write the query? We used WikiSQL: 80,654 questions over 24,241 Wikipedia tables, with train, dev and test on
disjoint tables, so the model is always tested on tables it has never seen.

## Framing it as translation
Columns become tokens `<c0>, <c1>, …`, so the model points at a column instead of spelling it. Example source and
target. One shared 8,000-piece BPE vocabulary.

## Building the Transformer
Walk through: scaled dot-product attention → multi-head → masks → encoder/decoder layers (post-norm) → shared
embedding/output weights. 7.58 M parameters. Mention the correctness checks (causal, padding, rows sum to 1,
weight sharing, LR schedule, gold round-trip 99.96%).

## Training
20 epochs, batch 64, Adam + warm-up schedule, label smoothing. ⟨training time⟩ on ⟨GPU⟩. Best dev loss ⟨…⟩ at
epoch ⟨…⟩. **Figure: loss curves.**

## Results
Table 3 (dev greedy, dev beam, test). Compare with the 36% LSTM baseline and >90% pretrained models.
Table 4 component accuracy: which part is hardest (usually WHERE values).

## What is the model looking at?
**Figure: cross-attention map.** When it writes `<c2>` it attends to `<c2> nationality` in the source;
⟨aligned_rate⟩% of generated column tokens attend most to the right column across 1,000 dev examples.

## Where it fails
Two or three cases from `results/samples.md`: wrong value span, wrong column among similar names,
missing condition. Why: no copy mechanism, trained from scratch on 56k examples.

## Takeaways
What we would try next within a bigger budget (copy/pointer attention, pretrained encoder), and the repo link.
