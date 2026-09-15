# Data

Data owns reporting. We also own the warehouse itself, but that is a vendor
product rather than a service in the service catalogue, so it does not appear
there.

Chen Wei and Freya Bakke are the engineers. Chen has been here longest and is
the person who knows why the revenue numbers in reporting are two hours behind
the ones in the ledger.

## The two-hour question

They are behind because reporting reads from the warehouse, and the warehouse is
loaded on a schedule. This is a known and accepted trade-off, not a bug, and it
is written down here so that it stops being raised as one.
