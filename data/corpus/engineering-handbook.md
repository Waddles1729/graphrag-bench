# Engineering handbook

How we work. This page is about process, not about any particular service, and
it is deliberately short.

## Reviews

Every change needs one approving review from someone who did not write it.
Changes to a service you do not own need an approval from someone who does.
Reviews are expected within one working day; if you cannot, say so rather than
letting it sit.

## Deploys

Continuous, behind a flag where the change is risky. The only standing freeze is
Payments' Friday afternoon rule. Everything else deploys when it is ready,
including on a Friday, and we have no evidence that this costs us anything.

## Incidents

Anyone can declare one. Declaring an incident that turns out to be nothing is
free; not declaring one that turns out to be something is expensive. The
severity scale runs from 1 (money or data is being lost right now) to 4 (a thing
is broken and nobody outside engineering can tell).

## Postmortems

Blameless, written within three working days, and read by the team that owns the
service. A postmortem with no action item is allowed and is sometimes the honest
answer: the system behaved as designed and the design was right.

## On-call

One week at a time. You are not expected to work your normal day after a night
that woke you. Take the morning.
