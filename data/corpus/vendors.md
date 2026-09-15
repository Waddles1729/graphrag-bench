# Third-party providers

Four vendors sit inside a request path or a nightly job. Anything else we buy
is a tool rather than a dependency and is not listed here.

**Stripe** handles card settlement. The ledger is the only service that talks to
it. Contract renews in March.

**Sift** provides external risk signals to fraud-scoring. Contract renews in
September. This is the one to watch: it is the only vendor whose slowness
degrades a customer-facing decision rather than failing it outright.

**Twilio** carries our SMS traffic, for notify. Renews in June. Email does not
go through Twilio and never has, despite the number of times that has been
assumed in a postmortem.

**Snowflake** is the data warehouse that reporting reads from. Renews in
January.

## Escalation

Every one of these has a named account manager; the contacts are in the
Operations space, not here, because they change more often than this page does.
