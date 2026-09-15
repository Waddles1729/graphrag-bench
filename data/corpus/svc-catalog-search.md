# catalog-search

Full-text and faceted search over the product catalogue. Read-only from the
customer's point of view; the index is rebuilt from the catalogue feed every
fifteen minutes.

It calls pricing to decorate results with a display price. That is its only
outbound dependency, and it is a soft one — if pricing is unavailable we serve
results without prices rather than serving nothing.

The index itself lives in `search-opensearch`.

## A caveat worth knowing

The price in a search result can be up to fifteen minutes stale, because it is
captured at index time and not at query time. checkout-api re-quotes, so the
customer is never charged the stale number, but they do sometimes see it.
