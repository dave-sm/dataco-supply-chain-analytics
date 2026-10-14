# Power BI Model Notes - DataCo Supply Chain Analytics
   
**Purpose:** Star-schema data model bridging the eight analytical notebooks into a unified BI layer for the six SCOR-aligned dashboards.   
**Grain:** Order line-item (159,763 rows in `FactOrders`).   
*Note:* The interactive `SmartSupplyChainSCOR.pbix` file is available upon request and is not included in the repo.

---

## Model architecture

Fourteen tables, organized into four groups.

### Measure tables

| Table | Purpose |
|-------|---------|
| `_Measures` | Container for all DAX measures. No columns, no data. Keeps measures organized and discoverable. |
| `WaterfallTable` | Calculated table with four labels (Reported Profit, Discount, Late Penalty, Adjusted Profit). Drives the profit waterfall visual via a `SWITCH` measure. |
| `LatePenaltyRate` | What-if parameter, series 0.00 → 0.10 step 0.01. Drives the sensitivity analysis from Notebook 4 (late-delivery penalty scenarios at 1%, 3%, 5%). |

### Fact tables

| Table | Grain | Rows | Source notebook |
|---|---|---|---|
| `FactOrders` | Order line-item | 159,763 | NB5 (`orders_silver_with_risk.parquet` + star-schema reshaping) |
| `FactBottleneck` | Market × Region × Mode | 50 (≥ 300-order floor) | NB6 (`bottleneck_rank.parquet`) |
| `FactWeeklyForecast` | Market × Category × Week | 24 | NB3 (`FactForecastWeekly.parquet`) |
| `LeadTimeMRM` | Market × Region × Mode | 61 (≥ 200-order floor) | NB6 (`lead_mrm.parquet`) |

### Dimension tables

| Table | Grain | Key |
|---|---|---|
| `DimDate` | Calendar day | `DateKey` (int64, YYYYMMDD) |
| `DimCustomer` | One row per customer | `CustomerKey` (int64) |
| `DimProduct` | One row per product | `ProductKey` (string, composite) |
| `DimGeo` | Market × Region × Country | `GeoKey` (string, composite) |
| `DimShippingMode` | One row per shipping mode (4 rows) | `ShippingModeKey` (string) |
| `CustomerSegments` | One row per customer | `CustomerKey` (int64) - joins to `FactOrders`, not `DimCustomer` |

---

## Relationships

| From | To | Key | Cardinality | Direction |
|---|---|---|---|---|
| `FactOrders` | `DimDate` | `DateKey` | many → one | single |
| `FactOrders` | `DimCustomer` | `CustomerKey` | many → one | single |
| `FactOrders` | `DimProduct` | `ProductKey` | many → one | single |
| `FactOrders` | `DimGeo` | `GeoKey` | many → one | single |
| `FactOrders` | `DimShippingMode` | `ShippingModeKey` | many → one | single |
| `FactOrders` | `CustomerSegments` | `CustomerKey` | many → one | **both** |
| `FactBottleneck` | `DimShippingMode` | `ShippingModeKey` | many → one | single |
| `FactBottleneck` | `DimGeo` | `GeoKey` | many → one | single |
| `LeadTimeMRM` | `DimShippingMode` | `shipping_mode` | many → one | single |
| `FactBottleneck` | `LeadTimeMRM` | `late_rate` | one → many | both, **inactive** |

**Note on the both-directions filter** between `FactOrders` and `CustomerSegments`. This enables the `Value_Share` measure to compute each service tier's share of total sales, correctly propagated through the customer dimension. Without bidirectional filtering, the measure would still work but the filter context wouldn't flow back from `CustomerSegments` to `FactOrders`.

**Note on the inactive relationship** between `FactBottleneck` and `LeadTimeMRM` on `late_rate`. Retained for future drill-through use; activate via `USERELATIONSHIP()` in DAX if needed.

---

## Data source

All tables are imported from Parquet files in `artifacts/`.

The Parquet files required are:

- `FactOrders.parquet`
- `FactBottleneck.parquet`
- `FactWeeklyForecast.parquet`
- `LeadTimeMRM.parquet`
- `DimDate.parquet`
- `DimCustomer.parquet`
- `DimProduct.parquet`
- `DimGeo.parquet`
- `DimShippingMode.parquet`
- `DimCustomerSegments.parquet` (loaded as `CustomerSegments`)

`WaterfallTable` and `LatePenaltyRate` are calculated tables and do not require external sources.

---

## Calculated columns

| Table | Column | Formula |
|---|---|---|
| `FactBottleneck` | `SegmentLabel` | `shipping_mode & ": " & market & " - " & order_region` - readable label for the bottleneck ranking visual. |

---

## Measures

All measures are stored in the `_Measures` table. See `dax_measures.txt` for the full formula list, grouped by theme.

---