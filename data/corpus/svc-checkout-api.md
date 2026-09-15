# checkout-api

The service that turns a basket into an order. It is the most heavily read
service we run and the one with the least tolerance for latency.

## What it calls

- **identity** — to establish who the customer is before anything else happens.
- **pricing** — for the authoritative quote. The price shown in search is
  advisory; this one is binding.
- **fraud-scoring** — for the accept / review / decline decision.
- **ledger** — to post the authorisation once the order is accepted.

The calls to pricing and fraud-scoring are made in parallel. identity is on the
critical path and cannot be, which is the single largest contributor to p99.

## Data

Orders live in `orders-postgres`. Nothing else writes to that database.

## Degraded behaviour

If fraud-scoring does not answer within 400ms we take the order and queue it for
manual review. If pricing does not answer, we fail the checkout outright — we
will not guess at a price.
