## Correctness checks

Model: random initialisation (no checkpoint yet), eval mode (dropout off), tolerance 1e-05. Batch: first 8 dev examples.

- **Causal mask:** PASS. Changing the last decoder input token: max |Δ| at earlier positions = 0.00e+00 (at the changed position itself: 2.76e+00).
- **Padding mask:** PASS. Appending 25 `<pad>` tokens to every source: max |Δ| of the output logits = 3.10e-06.
- **Attention rows sum to 1:** PASS. Over all 9 attention modules, max |row sum over unmasked keys − 1| = 2.38e-07; max weight on a masked key = 0.00e+00.
- **Weight sharing:** PASS. `model.generator.weight is model.shared.emb.weight` → True; encoder and decoder InputLayers hold the same TokenEmbedding → True. Trainable parameters: 7,577,600.
- **Learning rate:** PASS. Peak 9.882e-04 at step 4000; lr/step constant to 2.1e-16 for steps 1–4000 (linear rise); lr·√step constant to 2.2e-16 for steps 4001–20000 (step^-0.5 decay). Plot: `results/figures/lr_schedule.png`.
- **Gold round-trip:** PASS. Official evaluator on parsed gold dev targets: execution 99.96%, logical form 99.96%, parse failures 0.00%. Residual errors come from values that SentencePiece changes irreversibly (NFKC normalisation, <unk> that cannot be matched in the question), not from the parser.

**6/6 checks passed.**
