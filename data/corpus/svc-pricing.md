# pricing

Works out what a given customer should pay for a given item right now:
list price, promotions, regional adjustments and stock-driven markdowns.

The stock-driven part is why pricing depends on warehouse-sync — a line that is
nearly out of stock stops being discounted. That is the only service pricing
calls.

Rules are held in `pricing-postgres` and the computed quote is not stored at
all; it is recomputed on every request, deliberately, so that there is exactly
one code path and no cache to go stale.

## Two callers, two contracts

catalog-search asks for an advisory price and will accept a slow or missing
answer. checkout-api asks for a binding quote and will not. The same endpoint
serves both, distinguished by a `binding=true` parameter, which in hindsight
should have been two endpoints.
