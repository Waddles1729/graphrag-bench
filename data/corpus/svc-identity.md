# identity

Accounts, sessions, device fingerprints and contact preferences. Identity is the
closest thing we have to a single point of failure, which is why it has no
dependencies of its own: it calls nothing.

Data lives in `identity-postgres`. Sessions are in the same cluster rather than
in a cache, which is a decision we revisit about once a year and have so far
always kept, because a session that survives a restart is worth more to us than
the milliseconds it costs.

## Who depends on it

Rather than list them here — the list goes stale — check the service catalogue.
The honest summary is: most things.

## Rate limits

Identity will shed load before it falls over. A caller that exceeds its quota
gets 429s, and the quota is per-service, not per-instance. If you are seeing
429s, the fix is a quota change and not a retry loop.
