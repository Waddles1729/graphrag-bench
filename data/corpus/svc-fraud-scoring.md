# fraud-scoring

Scores an order for fraud risk and returns accept, review or decline. It is
called synchronously by checkout-api and has a hard 400ms budget.

Two things feed the score. The first is our own feature store, built from
device and order history, which we pull from identity — identity holds the
device fingerprints and we have never been happy about that arrangement. The
second is Sift, a third-party risk provider, which contributes about a third of
the signal weight.

Features are cached in `risk-redis`. The cache is not durable and a cold start
costs us roughly ninety seconds of degraded accuracy.

## Failure modes

Sift being slow is more common than Sift being down. When either happens we
score on our own features only and flag the decision as partial, which is why
the review queue gets long during a Sift incident rather than during an outage
of ours.
