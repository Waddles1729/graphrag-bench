# Risk

Risk exists to keep fraudulent orders out without turning away good customers.
We run fraud-scoring, and that is deliberately the only service we own — the
team is small and the decision latency budget is tight enough that we would
rather do one thing properly.

Lena Fischer leads the team. Tom Okafor is the other engineer and owns the
feature store.

We publish our false-positive rate weekly. When it moves, it is usually because
an upstream signal changed, not because our model did.
