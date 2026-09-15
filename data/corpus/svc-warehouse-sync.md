# warehouse-sync

Keeps our idea of stock in step with the warehouse management system. It is a
batch job wearing a service's clothes: an API for reads, and a nightly
reconciliation that does the real work.

It has no dependencies on other services. It talks to the WMS over SFTP, which
is as unpleasant as it sounds and is the reason the reconciliation is nightly
rather than continuous.

Stock levels are written to `stock-postgres`.

## The nightly job

Starts at 02:00 UTC and normally finishes by 02:40. If it is still running at
04:00 it will not have finished, and somebody should look at it rather than
waiting. A failed run leaves yesterday's numbers in place; it does not leave
partial ones.
