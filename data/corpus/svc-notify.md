# notify

One way to send a message to a customer, so that no other service has to know
about email templates, SMS gateways or quiet hours.

notify depends on identity for contact preferences — address, number, locale
and whether the customer has opted out of a category. For SMS it depends on
Twilio. Email goes out through our own relay, which is not a service in the
catalogue and not something anyone wants to own.

Queued messages sit in `notify-kafka` until they are delivered or expire.

## Ordering

We do not guarantee ordering between channels. An SMS can and does arrive
before the email that was queued first. If ordering matters to you, send one
message.
