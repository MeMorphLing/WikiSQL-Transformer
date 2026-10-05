#!/usr/bin/env bash
# End-to-end reproduction: data -> tokenizer -> checks -> train -> dev eval -> test (once) -> report.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/setup_data.sh
(cd starter && python data_prep.py && python tokenizer.py && python check_starter.py)   # Task 1.1-1.2
python data_report.py                      # Task 1.3-1.4  (Table 1, PE heat-map)
python checks.py                           # correctness checks (random init + gold round-trip)
python train.py                            # Task 3        (20 epochs, best dev-loss checkpoint)
python checks.py                           # re-run the checks on the trained checkpoint
python evaluate_model.py dev               # Task 4-5 on dev (greedy + beam, components, attention, samples)
python evaluate_model.py test              # Task 5.1 test, ONCE, with the decoding chosen on dev
bash scripts/check_compliance.sh
echo "Done. See results/tables.md and results/figures/."
