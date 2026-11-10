# Reset A21 control in the matched CISC experiment

This is the reset-state control evaluated on clean candidate pools with configured fault budget 3. Reset credibility and sender confidence are 0.5. Required binary receiver scores are the normalized CISC-P probabilities.

1. An empty pool abstains. Any missing/invalid required score on a valid candidate returns original-pool majority before the gate.
2. Configured fault budget zero returns majority. At budget 3, at most three valid candidates cause abstention.
3. With reset state, credibility spread and minority excess triggers are inactive, leaving the top-support trigger at 0.72. An inactive gate returns majority. An active gate retains all valid seats; a unique plurality also returns directly. Every top-count tie activates the support trigger.
4. On a top-count tie, sum `0.5 * q_i` over every observed class, including classes below the top count. Equal class sums use first occurrence. Total weight at most 1e-12 returns the first top-count class.

The standard-library replay implements this equivalent reset-state decision tree and verifies every stored decision. Persistent-state variants and attacked-history experiments are outside the current manuscript.
