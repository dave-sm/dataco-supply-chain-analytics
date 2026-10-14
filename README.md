# DataCo Smart Supply Chain Analytics
### An end-to-end SCOR pipeline with a deployed late-delivery risk scorer

An end-to-end supply chain analytics project built on the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) - 180,519 raw transactions spanning 2015-2018 - structured around the **SCOR framework** and shipped with a **live interactive web application**.

**Stack:** Python (Kaggle + local), LightGBM, scikit-learn, SHAP, Streamlit, Power BI.


## Quick links

> **Live Demo:** [DataCo Supply Chain Analytics](https://dataco-supply-chain-analytics-dsm.streamlit.app/)   
> **Deep dive:** see sections 4-8 below.

## Headline findings

- **Late delivery rate: 54.9%.** The network breaches its delivery SLA on more than half of all shipments. Approximately **$15.9M of revenue is associated with late-flagged orders**.
- **The cause is SLA calibration, not logistics capability.** The network delivers in a tight 2-6 day window uniformly across all 23 corridors. What varies is the *promise* attached to each shipping mode.
- **First Class is the riskiest mode.** 95.3% of First Class shipments are late despite delivering consistently at 2 days. The SLA is implicitly 1 day; the network delivers in 2. Standard Class, with a more honest promise, achieves 38.1%.
- **Discount leakage is the largest margin threat (\$3.26M),** late-delivery penalties are second (\$477K). The deployed model addresses the second channel directly.
- **Calibrated predictions.** The deployed model's mean predicted probability (0.5492) matches the observed late rate (0.5488). Every decile is within ±3.5 percentage points.

## The application

A Streamlit web app exposes the late-delivery risk model as a live scoring tool. A user configures an order (market, shipping mode, category, scheduled days, etc.) and receives:

- Predicted probability of lateness.
- Risk tier (Top 1%, Top 5%, Top 10%, Top 30%, Bottom 50%).
- A decision (Flag / Standard) using a percentile-based threshold.
- Context: the model's calibration, base rate, and the enterprise-level KPI framework.

**Four tabs:**

1. **Live Risk Scorer:** Interactive prediction with adjustable intervention threshold.
2. **Business Context:** SCOR KPI baseline, profit leakage decomposition, shipping mode performance.
3. **Demand Planning:** Read-only weekly forecast from Notebook 3.
4. **About the Model:** Model card, feature importances, leakage audit, caveats.

## Model card

| Field | Value |
|-------|-------|
| Target | `late_delivery_risk` (1 = late, 0 = on-time) |
| Model | LightGBM (800 trees, lr 0.03) |
| Training set | 127,810 orders |
| Test set | 31,953 orders |
| Base rate | 54.9% |
| **ROC-AUC** | **0.7706** |
| **PR-AUC** | **0.8345** |
| **Calibration** | Mean predicted 0.5492 vs observed 0.5488 |
| **Max decile calibration error** | ±3.5 percentage points |
| Features | 25 (14 categorical, 11 numeric) |
| Data horizon | 2015-01 to 2018-01 |
| Selected model rationale | Tied on ROC-AUC and precision with LogReg; selected for superior calibration. |

**Leakage audit.** 30 columns were excluded across five categories: target, post-outcome, realized financial outcomes, identifiers, and confirmed synthetic location coordinates. See notebook 5 Leakage Audit (notebooks/ `05_operational_risk_source_make_deliver_sc.ipynb`) for the full audit.

## Repository structure

```
dataco-supply-chain-analytics
    app.py                           # Streamlit interface
    inference.py                     # Model loading + prediction API
    requirements.txt                 # Pinned dependencies
    .streamlit/
        config.toml                  # Theme and server config
    model-artifact/                  # Deployed model + schema + metrics
        deploy_model.joblib          # 3 MB LightGBM pipeline
        deploy_feature_schema.json
        deploy_metrics.json
        deploy_importances.parquet
    artifacts/                       
        ...                          # Governed data outputs
    notebooks/                       # 8 analytical notebooks (00-09)
        00-project-setup-and-data-dictionary.ipynb
        01-data-quality-and-cleaning.ipynb
        02-kpi-scor-baseline.ipynb
        03-demand-forecasting-plan-deliver.ipynb
        04-financial-impact-measure-return.ipynb
        05-operational-risk-source-make-deliver.ipynb
        06-global-logistics-deliver.ipynb
        07-customer-centric-ops-plan-deliver-return.ipynb
        08-powerbi-export-gold-star-schema.ipynb
        09-embedded-power-bi-dashboards.ipynb
    power-bi/                       # BI model notes and DAX
        model_notes.md
        dax_measures.txt
    data_dictionary.csv
```

## Running locally

- git clone https://github.com/dave-sm/dataco-supply-chain-analytics   
- cd dataco-supply-chain-analytics   
- python -m venv .venv     
- pip install -r requirements.txt   
- streamlit run app.py    
 
<details>
<summary><b>Data pipeline</b></summary>

- Raw DataCo CSV (180,519 × 53)
    - Silver layer cleaning, dedup, feature derivation (159,763 × 55).   
        - Gold: SCOR KPI baseline (12 metrics).
        - Gold: weekly demand by market (234 rows).
        - Modeling layer: leakage audit, preprocessing, training.
            - Late-delivery classifier (LightGBM, ROC-AUC 0.771).
            - Risk-enriched silver (159,763 × 56).
                - Streamlit app: live scoring.
    - Power BI: 6-table star schema + 5 analytical tables.

</details>

## What this project demonstrates

- **Grounding in operational context:** Every model connects to a SCOR process and a specific business decision.
- **Leakage discipline:** An initial version of the risk model leaked post-outcome information. The corrected version uses only pre-dispatch signals. The audit is documented in Notebook 5 and reflected in the deployed model card.
- **Calibration over accuracy:** The chosen model was selected for calibration quality, not ROC-AUC, because the app surfaces raw probabilities. LogReg and LightGBM were tied on ranking; LightGBM was materially better calibrated.
- **End-to-end thinking:** From raw CSV → governed artifacts → deployed interface → BI-ready star schema.

## Deep dive

The sections below document the full analytical pipeline. They are long by design - the sections above give the shorter version; the sections below provide the eight-notebook narrative.

### 1) Project Summary

This project converts transactional supply chain data into a decision-ready intelligence layer spanning five analytical pillars:

- Demand forecasting.
- Service reliability and operational risk.
- Logistics bottleneck diagnostics.
- Financial profitability and leakage attribution.
- Customer-centric service tier design.

The work is organised end-to-end using the **SCOR (Supply Chain Operations Reference)** model, ensuring each analysis serves a defined operational audience - planning teams, risk and reliability engineers, and senior decision-makers - rather than existing as a standalone exercise.

|Analytics Focus |SCOR Phase(s) |Business Value Demonstrated|
|----------------|--------------|---------------------------|
|Demand Forecasting |Plan / Deliver |Capacity planning, inventory positioning, service-level risk anticipation.|
|Operational Risk Modeling |Source / Make / Deliver |Late-delivery probability scoring, interpretable risk drivers, proactive intervention triage.|
|Global Logistics Diagnostics |Deliver |Bottleneck identification, mode-level SLA calibration, corridor prioritisation.|
|Financial Impact Analysis |Measure / Return |Profit concentration mapping, leakage decomposition, margin recovery recommendations.|
|Customer-Centric Operations |Plan / Deliver / Return |RFM segmentation, service equity diagnostic, differentiated service tier policy.|

The **central cross-notebook finding**: late delivery in this supply chain is an **SLA design problem**, not a logistics capability problem. The network delivers consistently (median 3 days, P90 6 days across all corridors), but late rates diverge sharply by shipping mode because service promises are calibrated more aggressively than the network's demonstrated delivery physics. The primary operational recommendation is **mode-level SLA recalibration**, not infrastructure investment, carrier renegotiation, or regional restructuring.

### 2) Key Business Questions

**Demand Forecasting (Plan / Deliver)**

- Where is demand increasing or decreasing by Market × Category?
- How volatile is demand, and when do simple baselines outperform machine learning?
- Which segments justify a dual-forecast approach (heuristic baseline + ML directional signal)?

**Financial Impact (Measure / Return)**

- Which markets, categories, and customer segments create versus destroy profit after operational leakage?
- How much profit leakage is attributable to discounts, late-delivery penalties (proxy), and returns/cancellations (proxy)?
- What is the ROI of discount governance versus service reliability improvement?

**Operational Risk (Source / Make / Deliver)**

- What drives late delivery risk, and can it be predicted before fulfillment?
- Which shipments warrant proactive intervention?
- How do shipping mode, order execution workflows, and order characteristics interact to produce service failure?

**Global Logistics (Deliver)**

- Where are the operational bottlenecks across regions, corridors, and shipping modes?
- Does geography explain variation in lead time or late-delivery exposure?
- What is the most efficient intervention to reduce late-delivery risk?

**Customer-Centric Operations (Plan / Deliver / Return)**

- Do high-value customers receive differentiated service, or is fulfillment uniform across segments?
- Does late delivery exposure correlate with reduced repeat purchasing (proxy)?
- What service tiers should govern shipping mode allocation, recovery policy, and fulfillment prioritisation?

### 3) Solution Overview

#### A) Data Engineering and Quality (SCOR Enable)   
*Notebooks: 00, 01.*

- **Programmatic data dictionary** capturing data types, missingness, cardinality, and example values for every feature.
- **Silver layer construction:** Schema standardisation (snake_case, date parsing, type enforcement), deduplication (11.5% duplicate rate addressed distinguishing legitimate order-item expansion from structural join inflation), and feature derivation (`delivery_days`, `shipping_delay`, `processing_time`, `profit_margin`).
- **Outlier analysis:** IQR-based detection across shipping delay (31,656 records), profit per order (16,978), sales per customer (1,639), and discount fields. All outliers are retained, most represent meaningful operational or financial behaviour.
- **SCOR KPI baseline:** Current-state performance scorecard spanning Plan (demand volatility CV ≈ 0.22), Source/Make (avg processing time ≈ 3.47 days), Deliver (55% late rate, median 3 days, P90 6 days), Return (≈ 2.2% return rate, ≈ \$73K impact), and Measure (≈ \$29.1M sales, ≈ \$3.48M reported profit, ≈ \$15.9M late-delivery value-at-risk).

- **Gold layer persistence:** Weekly demand by market and SCOR KPI dictionary exported for downstream reuse.

**Output:** 159,763 records × 53 features in the analytics-ready silver dataset.

#### B) Demand Forecasting System (Plan / Deliver)   
*Notebook: 03.*

**Approach:** Aggregate → Segmented → Cross-Validated → Forward-Projected.

- Weekly demand series constructed at three grains: Aggregate, Market (6 regions), and Market × Category.
- **Two-model architecture:** Moving Average baseline (stability anchor) + LightGBM (momentum-sensitive directional signal).
- Time-series cross-validation with sMAPE and MAE.
- Market volume audit classifies regions into Primary (LATAM, Europe), Growth (Pacific Asia, USCA), and Niche (Africa) tiers.

**Key findings:**

|Segment |Best performer |sMAPE |Interpretation|
|--------|---------------|------|--------------|
|Aggregate |12-week MA |15.55% |Short-term momentum dominates; ML constrained by aggregation dilution.|
|LATAM Cleats |4-week MA |11.10% |Short runway (40 weeks); high stochastic volatility favours smoothing.|
|Europe Cleats |LightGBM |32.49% |Longer runway (57 weeks); ML captures seasonal regime shift.|

The **dual-forecast planning envelope** is deployed: MA for conservative capacity buffers, ML for agility overlay. In Europe, the ML model anticipates the historical Q4 contraction (~58% demand decline in weeks 40-52) that the MA benchmark systematically overestimates.

**Planning translation layer** converts forecast revenue into shipment volume (\~32 shipments/week for LATAM | Cleats), workforce load (\~0.8 FTE), and capacity strain risk indicators.

**Strategic verdict:** Segmentation-first forecasting. Model selection should be calibrated to segment-level data depth and volatility structure, not applied uniformly.

#### C) Financial Impact Analysis (Measure / Return)   
*Notebook: 04.*

**Approach:** Multi-grain profit concentration → Leakage waterfall → Sensitivity analysis → Margin recovery recommendations.

Profit concentration analysis across four grains: Market × Region (23 corridors), Category, Customer Segment, and Individual Customer. Three economic archetypes emerge: Profit Creators (East Africa, Canada), Profit-Neutral High-Scale (East of USA, South America), and Profit Destroyers (US Center, Southern Europe, Southeast Asia).

**Enterprise Profit Waterfall:**

|Component |Value ($) |% of Reported Profit|
|----------|------------|--------------------|
|Gross Sales |29.05M |  |
|Reported Profit |3.48M	| 100% |
|Discount Leakage |-3.26M |-93.7% |
|Late Delivery Penalty (3% proxy) | -476.9K | -13.7% |
|Return/Cancel Impact | -73.1K | -2.1% |
|**Adjusted Profit** | -259.8K | -7.5% |

**Key insight:** Discount leakage alone consumes nearly the entirety of reported profit. Discount rates are near-identical across all markets and categories (11.0-12.5% of sales), indicating a centralised pricing policy.

**Late Delivery Value-at-Risk:** Late rates cluster between 53-56% across all major regions. More than half of enterprise revenue ($15.9M) is fulfilled under compromised service conditions. Western Europe and Central America carry the highest absolute exposure.

**Sensitivity analysis** on the late-delivery penalty rate (1% / 3% / 5%) reveals three regional archetypes:

- **Margin-Resilient:** Canada, East Africa - profitable at all three rates.
- **Tipping-Point:** South America, Southern Europe, Southeast Asia, Oceania - profitable at 1%, negative at 3%.
- **Structurally Fragile:** Caribbean, West USA, Central Africa, West Asia, Eastern Europe - negative at all three rates.

**Margin recovery recommendations (ordered by expected impact):**

1. **Reduce discount leakage by 10%** (\~\$326K recovery): enterprise-wide 1.1 pp reduction.
2. **Reduce late delivery rates by 20%** in tipping-point regions (\~\$43K recovery).
3. **Monitor returns**; no policy change warranted at current leakage levels.

**Combined impact of levers 1 and 2:** adjusted profit moves from -\$260K to approximately +\$109K without demand growth or capital investment.

#### D) Operational Risk Model (Source / Make / Deliver)   
*Notebook: 05 - the deployed model.*

**Approach:** Dual-model binary classification with comprehensive interpretability layer, calibrated for pre-dispatch scoring.

**Target:** `late_delivery_risk` (54.88% positive class - structurally systemic, not a rare event).

**Leakage audit.** 30 columns were excluded across five categories:

|Category |Count |Columns |
|---------|------|--------|
|Target | 1 |`late_delivery_risk`|
|Post-outcome | 6 | `delivery_status`, `actual_delivery_date`, `shipping_delay`, `days_for_shipping_real`, `processing_days`, `order_status`
|Realized financial outcome | 6 |`benefit_per_order`, `profit_margin`, `order_profit_per_order`, `order_item_profit_ratio`, `order_item_total`, `sales`|
|Identifiers and timestamps | 15 |`customer_id`, `order_id`, `order_item_id`, `product_card_id`, `PII fields`, `order_date`, `shipping_date`
|Synthetic noise |2 |`latitude`, `longitude`|

**The latitude/longitude exclusion.** In the DataCo dataset these columns are not real geographic coordinates. A region-level audit confirmed that the median coordinate for every (`market`, `order_region`) pairing clusters within a single degree of (33, -77) - the coast of North Carolina. Cities as diverse as Kinshasa, Sydney, Tokyo, and Madrid all share coordinates near that point. The two columns act as high-cardinality noise; tree models fit them heavily. Removing them cost the model 1.7 points of ROC-AUC but improved decile-level calibration by 60%.

**Model performance (corrected, pre-dispatch-only features):**

|Metric |Logistic Regression |LightGBM|
|-------|--------------------|--------|
|ROC-AUC | 0.7713 | 0.7706 |
|PR-AUC | 0.8317 | 0.8345 |
|Accuracy @ 0.5 | 71% | 71% |
|Late precision @ 0.5 | 0.82 | 0.83 |
|Late recall @ 0.5 |0.60 | 0.58 |
|Mean predicted | 0.5136 | 0.5492 |
|Max decile calibration error | 0.0784 | 0.0352 |

**Model selection.** The two models are tied on ROC-AUC and on operational precision. They diverge sharply on calibration:

- Logistic Regression systematically under-predicts: mean predicted 0.5136 vs observed 0.5488, with a max single-decile error of 7.8 pp.
- LightGBM is well-calibrated: mean predicted 0.5492 vs observed 0.5488, max error 3.5 pp.

Because the Streamlit app surfaces raw probabilities directly to users, **calibration is the deciding criterion**. LightGBM is selected as the deployed model. The logistic baseline is retained as a documented alternative.

**Interpretability - what drives predictions (by mean |SHAP|):**

|Rank |Feature |Mean |SHAP|Direction |
|-----|--------|-----|----|----------|
|1 |`shipping_mode_First Class` | 1.013 |First Class active → risk up sharply.|
|2 |`shipping_mode_Standard Class` | 0.674 |Standard active → risk down. |
|3 |`type_TRANSFER` | 0.354 |TRANSFER active → risk down. |
|4 |`days_for_shipment_scheduled` | 0.151 |Tighter window → risk up. |
|5 |`type_PAYMENT` | 0.053 |PAYMENT active → mild risk down. |
|6 |`type_DEBIT` | 0.052 |DEBIT active → mild risk up. |
|7 |`shipping_mode_Same Day` | 0.051 |Same Day active → mild risk up. |
|8 |`shipping_mode_Second Class` | 0.047 |Second Class active → mild risk up. |
|9 |`type_CASH` | 0.011 |CASH active → mild risk up. |
|10 |`order_item_discount` | 0.011 |Higher discount → mild risk up. |

Shipping mode dominates. The model independently rediscovers the Notebook 6 SLA calibration finding.

**Threshold selection.** The deployed model uses a percentile-based classification rule, rather than a fixed 0.5:

|Percentile	Cutoff |Precision (late) |Recall (late) |F1|
|------------------|-----------------|--------------|--|
|Top 10% |0.9981 |1.000 |0.182 |0.308|
|Top 20% |0.8026 |0.938 |0.342 |0.501|
|Top 30% |0.7160 |0.886 |0.484 |0.626|
|Top 50% |0.4113 |0.747 |0.681 |0.712|

The app exposes this threshold as a user input, defaulting to **top 30% as the balanced operating point**.

**Output:** `orders_silver_with_risk.parquet` - risk-enriched Silver+ layer with `late_risk_score` appended for Power BI ingestion.

#### E) Global Logistics Analytics (Deliver)   
*Notebook: 06.*

**Approach:** Six-stage diagnostic elimination sequence narrowing the search space for late-delivery root cause.

- **Stage 1 - Lead-Time Engineering:** `delivery_days` is calculated from shipment and delivery timestamps with fallback logic. Distribution is compact: 0-6 day corridor, mode 2 days (31.4%), median 3 days, mean ≈ 3.5 days, CV 0.46. No extreme outliers.

- **Stage 2 - Global Baseline Diagnostic:** Median 3 days, P90 6 days, 54.88% late rate. Predicted risk (0.5490) aligns with observed late rate (0.5488).

- **Stage 3 - Market × Region Segmentation (23 corridors):** Null result. Average lead time varies by ±0.06 days. Median and P90 fixed at 3 and 6 days everywhere. Late rates fall within a 53.1-56.4% band. Geography is eliminated as an explanatory variable.

- **Stage 4 - Shipping Mode Segmentation:** Decisive finding:

|Mode |Avg days |P90 |Late rate|
|-----|---------|----|---------|
|Same Day | 0.48 | 1 | 45.7% |
|First Class | 2.00 | 2 | 95.3% |
|Second Class | 3.99 | 6 | 76.8% |
|Standard Class | 3.99 | 6 | 38.1% |   

<div style = "padding-left: 40px;">
Second Class and Standard Class share identical physical delivery profiles but diverge by 38.7 percentage points in late rate. The gap is entirely attributable to different SLA thresholds applied to identical transit performance.
</div>   
<br>

- **Stage 5 - Market × Region × Mode (61 segments):** Mode effects are globally uniform. Second Class produces 75-78% late rates in every region. Standard Class produces 35-42% late rates in every region.

- **Stage 6 - Bottleneck Ranking (50 segments):** Composite score integrates P90 tail exposure, late rate, and log-transformed volume. Two-tier architecture: Critical (48 segments, predominantly Second Class) and Elevated (2 segments, Standard Class).

**Strategic conclusion:** The network's fulfillment physics are sound. The problem is that SLAs are calibrated more aggressively than the network's demonstrated delivery capability. **The primary operational recommendation is mode-level SLA recalibration:**

- **Second Class:** Align SLA thresholds with Standard Class levels, reduces late rate from ~76% to ~38% with zero operational change.
- **First Class:** Shift promise window from implied 1-day to demonstrated 2-day delivery capability, eliminates the 95.3% late rate.
- **Deployment sequence:** Volume-prioritised by corridor, starting with Western Europe and Central America.

**The network is not broken, it is over-promised. The fix is calibration, not capacity.**

**Exported decision tables:** `mode_performance.parquet`, `bottleneck_rank.parquet`, `lead_mrm.parquet`.

#### F) Customer-Centric Operations (Plan / Deliver / Return)   
*Notebook: 07.*

**Approach:** RFM-inspired behavioural segmentation enriched with operational risk scores.

**Customer base:** 20,652 unique customers. Mean 3.18, Median 3 orders, Median recency 159 days. Monetary value right-skewed.

**RFM-derived segments:**

|Segment |Count |% of base |Strategic meaning|
|--------|------|----------|-----------------|
|Others | 11,951 | 57.9% |Mid-tier baseline cohort.|
|High-Value At-Risk | 7,562 | 36.6% |Revenue-critical, service-exposed.|
|Loyal/Active | 623 | 3.0%	|Stable growth core.|
|Dormant/At-Risk | 516 | 2.5% |Churn-sensitive recovery cohort. |

**Service equity diagnostic (central finding):**

Delivery performance is structurally uniform across all customer segments, and that uniformity is the problem. Median lead time (3 days), P90 (6 days), and late rates (~55%) are near-identical regardless of customer value. This reflects near-identical shipping mode allocation across segments.

**High-value customers receive no differentiated fulfillment treatment.** The supply chain is economically dependent on the High-Value At-Risk segment but does not service it differently. The network operates on volume-driven logistics, not value-driven logistics.

**Post-late repurchase proxy:** Customers with late exposure are the most frequent, highest-spending accounts (median 4 orders vs 1). Customer-level correlation between late rate and order count: **-0.002, effectively zero.** Late delivery is a margin risk, not a revenue risk.

**Differentiated service tier recommendations:**

|Tier |Segment |Strategy|
|-----|--------|--------|
|Priority Protection |High-Value At-Risk |Mode reassignment from Second/First Class to Standard Class.|
|Growth Priority |Loyal/Active |Mode allocation review.|
|Standard Service |Others |Cost-optimised, standardised fulfillment.|
|Recovery / Low Priority |Dormant/At-Risk |Efficient modes; selective reactivation.|

The correction does not require the network to move faster. It requires the right customers to be assigned to the right shipping modes.

#### G) Power BI Export: Gold Star Schema   
*Notebook: 08.*

The star schema bridges analytical outputs from all upstream notebooks into a unified Power BI data model.

**Core Star Schema (6 tables):**

|Table |Grain |Description|
|------|------|-----------|
|`FactOrders`|Order line-item (159,763 rows, 21 columns) |Transactional measures, engineered lead-time metrics, risk scores.|
|`DimDate`|Calendar day |Year, quarter, month, week attributes.|
|`DimCustomer` |One row per customer |Demographics, location, CustomerKey.|
|`DimProduct` |One row per product |Category and department hierarchies.|
|`DimGeo` |Market × Region × Country |Geographic hierarchy with composite GeoKey.|
|`DimShippingMode` |One row per mode (4) |Mode-level KPIs + SLA calibration fields.|

**Upstream analytical tables (5 tables):**

|Table |Source |Grain|
|------|-------|-----|
|`DimCustomerSegments` |NB7 |One row per customer.|
|`FactWeeklyForecast` |NB3 |Market × Week|
|`FactBottleneck` |NB6 |Market × Region × Mode (50 segments)|
|`LeadTimeMRM` |NB6	|Market × Region × Mode (61 segments)|
|`SCOR KPI Baseline` |NB1-2 |SCOR phase level.|

**Three analytical workflows are enabled by the schema:**

1. **SLA Calibration:** `DimShippingMode` carries observed lead-time distributions, late rates, and implied SLA thresholds. Powers the *Promise vs. Physics* visualisation.
2. **Intervention Prioritisation:** `FactBottleneck` provides a sequenced intervention shortlist classified into Critical and Elevated tiers.
3. **Customer-Centric Service Allocation:** `DimCustomerSegments` joins to `FactOrders` for cohort analysis without recomputing segmentation logic.

### 4) Power BI Deliverables

**Star Schema relationships:**

- `DimDate` → `FactOrders` (DateKey).
- `DimCustomer` → `FactOrders` (CustomerKey).
- `DimProduct` → `FactOrders` (ProductKey).
- `DimGeo` → `FactOrders` (GeoKey).
- `DimShippingMode` → `FactOrders` (ShippingModeKey).
- `DimCustomerSegments` → `FactOrders` (CustomerKey).

**Six dashboard pages:**

- **Strategic Overview:** Core KPIs, leakage decomposition, executive-level supply chain health.
- **Demand and Forecasting:** 12-week outlook, model comparison, forecast-vs-actual.
- **Financial Performance and ROI:** Profit waterfall, discount diagnostics, penalty sensitivity, margin recovery scenarios.
- **Operational Risk:** Risk score distribution, high-risk monitoring, SHAP driver analysis, scorecards.
- **Global Logistics and Bottlenecks:** Lead time distributions, bottleneck ranking, Promise vs. Physics SLA Gap.
- **Customer Service Tiers:** Value concentration, service equity dashboard, tier policy recommendations.

### 5) Assumptions and Proxies

The DataCo dataset does not contain full cost accounting (carrier invoices, labour cost, returns processing cost). Transparent proxy assumptions are introduced and validated through sensitivity analysis:

|Assumption |Proxy |Sensitivity treatment|
|-----------|------|---------------------|
|Late delivery penalty |3% of sales for late-flagged orders. |Stress-tested at 1% and 5%.|
|Return/cancellation impact |Profit on orders with status flags: CANCELLED, RETURN, REBATE, SUSPECTED_FRAUD. |Monitored; not material at current levels.|
|Capacity metrics |Derived from average order value, units-per-order ratio, packing density proxy. |Planning-grade proxies.|
|Lead time |`actual_delivery_date - shipping_date` (administrative), `days_for_shipping_real` as fallback |Uncapped distribution is source of truth.|
|Repeat purchasing |Order frequency + post-late order placement |Interpreted directionally (r = -0.002).|
|Discount leakage |`order_item_discount` as direct reduction. |Validated - no rates exceed 100%.|
|SLA thresholds (implied) |Inferred from observed delivery distributions. |SLA gap computed from observed performance.|

### 6) Cross-Notebook Synthesis

|SCOR Layer |Notebook |Central Finding|
|-----------|---------|---------------|
|Plan |Demand Forecasting | Demand moderately volatile (CV ≈ 0.22) and forecastable. Dual-forecast envelope: MA for stability, ML for directional agility.|
|Source / Make / Deliver |Operational Risk |Late delivery structurally predictable (ROC-AUC 0.771). Shipping mode is the dominant driver. Calibration is the model selection criterion.|
|Deliver |Global Logistics |The network delivers consistently; late rates diverge by shipping-mode SLA design, not geography.|
|Measure / Return |Financial Impact |Discount leakage (\$3.26M) is the dominant profit destroyer. Late penalties (\$477K) are secondary but systemic.|
|Plan / Deliver / Return |Customer-Centric Ops |High-value customers receive no differentiated service. Late delivery is a margin risk, not a revenue risk.|

**The central finding:** Late delivery in this supply chain is an **SLA design problem, not a logistics capability problem**. The network is not broken, it is over-promised. The fix is calibration, not capacity.

### 7) How to Reproduce

**Kaggle:**

1. Add the DataCo dataset to your Kaggle notebook environment, or local environment.
2. Run notebooks sequentially (00 → 08). Gold-layer Parquet files are generated in /kaggle/working/.
3. Download all .parquet, .csv files for downstream use.

**Power BI:**

1. For model documentation and dax measures, see: 
    - `powerbi/model_notes.md`.
    - `powerbi/dax_measures.txt`.
5. For dashboard visualizations, see: `09-embedded-power-bi-dahsboards.ipynb`.

**Streamlit app:**

1. pip install -r requirements.txt
2. streamlit run app.py

### 8) Next Improvements

- **Time-aware model validation:** Train on earlier periods, validate on later periods.
- **Forecast prediction intervals:** Quantile regression or conformal prediction for confidence bands.
- **Cost-to-serve attribution:** Shipping cost proxies for true shipment-level profitability.
- **Causal churn analysis:** Replace the repurchase proxy with formal retention modelling if tenure data becomes available.
- **Exogenous feature enrichment:** Promotional calendars, holiday flags, pricing changes for the forecasting model.
- **Mode-level SLA simulation engine:** What-if projections of late-rate reduction under different SLA threshold scenarios.
- **Market × Category forecast expansion:** Extend the LATAM | Cleats framework to additional high-volume intersections.
- **Operational risk threshold optimisation:** Model the cost trade-off between false positives and false negatives to determine the economically optimal intervention threshold.

## Summary

Portfolio project demonstrating SCOR-aligned supply chain analytics: demand forecasting, operational risk modeling, logistics bottleneck diagnostics, financial leakage attribution, customer-centric service design, and a deployed Streamlit application, with full cross-notebook integration, transparent proxy assumptions, and BI-ready outputs.