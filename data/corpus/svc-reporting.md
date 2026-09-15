# reporting

Dashboards and scheduled extracts for everyone outside engineering. Finance,
Operations and the weekly business review all read from here.

reporting depends on Snowflake, which is where the modelled data lives. It also
reads directly from ledger for the two figures that have to be exact to the
penny and cannot wait for a warehouse load, and from catalog-search for product
metadata that never made it into the warehouse.

Nothing is stored by reporting itself; it is a query layer.

## Freshness

Everything sourced from Snowflake is up to two hours behind. The two figures
read from ledger are current. A dashboard that mixes them is therefore
internally inconsistent for up to two hours, which is why the revenue tile
carries a timestamp and the order-count tile does not.
