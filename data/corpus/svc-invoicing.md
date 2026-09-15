# invoicing

Produces invoices and credit notes for business customers, monthly, and chases
the ones that go unpaid.

It reads balances from ledger and sends the resulting documents through notify.
Those are its two dependencies. It does not talk to Stripe directly — everything
about settlement is the ledger's business, and invoicing only reports on it.

Generated documents are kept in `invoice-s3`.

## Timing

The monthly run starts on the first of the month at 06:00 UTC and takes about
forty minutes. It is idempotent: re-running it produces the same invoice numbers
and will not double-send.
