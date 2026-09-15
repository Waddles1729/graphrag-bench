# Platform

Platform builds the things every other team depends on. That makes us unusual
in one respect: we are almost never the team that notices an outage first.

| Service | What it is for |
| --- | --- |
| identity | Accounts, sessions, contact preferences |
| notify | Email, SMS and push delivery |

Both of the above are owned by Platform.

Marcus Bell is the engineering manager. Ravi Menon and Ingrid Holm are the
engineers; Ingrid wrote most of notify and Ravi has been on identity since it
was split out of the monolith.

## A standing request

Please do not add a hard dependency on identity for anything that has to work
while identity is down. We say this a lot. It is still true.
