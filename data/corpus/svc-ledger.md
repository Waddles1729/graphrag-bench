# ledger

Double-entry accounting for every movement of money. The ledger is append-only:
nothing is ever updated or deleted, and a correction is a new pair of entries.

It depends on identity, which resolves the actor on each entry — every posting
records who caused it, and "who" is an identity subject rather than a raw user
id. It also depends on Stripe, which is where settlement actually happens; the
ledger records what Stripe reports and reconciles nightly.

Entries are stored in `ledger-postgres`, on its own cluster, with a retention
policy of forever.

## Why it is slow to change

Every schema change to the ledger needs sign-off from Finance as well as from
Payments. Budget two weeks, not two days.
