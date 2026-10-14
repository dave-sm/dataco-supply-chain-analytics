"""
app.py
------
Streamlit web app for the DataCo late-delivery risk model.

Four tabs:
    1. Live risk scorer      - interactive prediction.
    2. Business context      - the SCOR KPI and financial backdrop.
    3. Demand planning       - read-only forecast from Notebook 3.
    4. About the model       - model card, metrics, feature drivers.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import inference


ARTIFACT_DIR = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "DataCo Late Delivery Risk",
    layout = "wide",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("DataCo Late Delivery Risk - Live Scoring")
st.caption(
    "Predict the probability that a shipment will breach its delivery SLA, "
    "using pre-dispatch signals only. The model is grounded in the SCOR "
    "Deliver process and calibrated against 159,763 historical orders "
    "(2015–2018)."
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_score, tab_context, tab_forecast, tab_about = st.tabs([
    "Live Risk Scorer",
    "Business Context",
    "Demand Planning",
    "About the Model",
])


# ---------------------------------------------------------------------------
# TAB 1 - Live Risk Scorer
# ---------------------------------------------------------------------------

with tab_score:
    schema = inference.get_schema()
    metrics = inference.get_metrics()
    cat_fields = schema["categorical"]
    num_fields = schema["numeric"]

    st.markdown(
        "Configure an order below. The model returns a live probability "
        "and a percentile rank against the test-set distribution."
    )

    # ---- Essential inputs (always visible) ----
    st.subheader("Essential inputs")
    c1, c2, c3 = st.columns(3)

    with c1:
        market = st.selectbox("Market", cat_fields["market"], index = cat_fields["market"].index("LATAM") if "LATAM" in cat_fields["market"] else 0)
        order_region = st.selectbox("Order region", cat_fields["order_region"], index = 0)
        shipping_mode = st.selectbox("Shipping mode", cat_fields["shipping_mode"], index = cat_fields["shipping_mode"].index("Standard Class") if "Standard Class" in cat_fields["shipping_mode"] else 0)

    with c2:
        type_ = st.selectbox("Payment type", cat_fields["type"], index = 0)
        customer_segment = st.selectbox("Customer segment", cat_fields["customer_segment"], index = 0)
        category_name = st.selectbox("Category", cat_fields["category_name"], index = 0)

    with c3:
        department_name = st.selectbox("Department", cat_fields["department_name"], index = 0)
        days_scheduled = st.slider(
            "Scheduled shipment days",
            min_value = int(num_fields["days_for_shipment_scheduled"]["min"]),
            max_value = int(num_fields["days_for_shipment_scheduled"]["max"]),
            value = int(num_fields["days_for_shipment_scheduled"]["median"]),
            step = 1,
        )
        order_item_quantity = st.slider(
            "Item quantity",
            min_value = int(num_fields["order_item_quantity"]["min"]),
            max_value = int(num_fields["order_item_quantity"]["max"]),
            value = int(num_fields["order_item_quantity"]["median"]),
            step = 1,
        )

    # ---- Advanced inputs (collapsed) ----
    with st.expander("Advanced inputs - geography, pricing, and discount"):
        a1, a2, a3 = st.columns(3)
        with a1:
            order_country = st.selectbox("Order country", cat_fields["order_country"], index = 0)
            order_state   = st.selectbox("Order state",   cat_fields["order_state"],   index = 0)
        with a2:
            order_city    = st.selectbox("Order city",    cat_fields["order_city"],    index = 0)
            customer_state = st.selectbox("Customer state", cat_fields["customer_state"], index = 0)
        with a3:
            product_name = st.selectbox("Product", cat_fields["product_name"], index = 0)
            customer_country = st.selectbox("Customer country", cat_fields["customer_country"], index = 0)

        b1, b2, b3 = st.columns(3)
        with b1:
            sales_per_customer = st.number_input(
                "Order value (USD)",
                min_value = float(num_fields["sales_per_customer"]["min"]),
                max_value = float(num_fields["sales_per_customer"]["max"]),
                value = float(num_fields["sales_per_customer"]["median"]),
                step = 10.0,
            )
            order_item_product_price = st.number_input(
                "Unit price (USD)",
                min_value = float(num_fields["order_item_product_price"]["min"]),
                max_value = float(num_fields["order_item_product_price"]["max"]),
                value = float(num_fields["order_item_product_price"]["median"]),
                step = 10.0,
            )
        with b2:
            order_item_discount = st.number_input(
                "Discount amount (USD)",
                min_value = float(num_fields["order_item_discount"]["min"]),
                max_value = float(num_fields["order_item_discount"]["max"]),
                value = float(num_fields["order_item_discount"]["median"]),
                step = 1.0,
            )
            order_item_discount_rate = st.slider(
                "Discount rate",
                min_value = float(num_fields["order_item_discount_rate"]["min"]),
                max_value = float(num_fields["order_item_discount_rate"]["max"]),
                value = float(num_fields["order_item_discount_rate"]["median"]),
                step = 0.01,
            )
        with b3:
            product_price = st.number_input(
                "Product list price (USD)",
                min_value = float(num_fields["product_price"]["min"]),
                max_value = float(num_fields["product_price"]["max"]),
                value = float(num_fields["product_price"]["median"]),
                step = 10.0,
            )

    # ---- Derive fields the user does not see ----
    # Pull the modal product for the selected category, so product_image and
    # product_name are always populated with realistic values.
    product_image = "http://images.acmesports.sports/placeholder"
    category_id           = 2
    department_id         = 2
    product_category_id   = 2
    product_status        = 0

    # ---- Percentile threshold selector ----
    st.markdown("---")
    threshold_label = st.radio(
        "Intervention threshold",
        options = [
            ("Top 10% - highest precision", 0.90),
            ("Top 20% - balanced precision", 0.80),
            ("Top 30% - recommended", 0.70),
            ("Top 50% - broad coverage", 0.50),
        ],
        format_func = lambda x: x[0],
        index = 2,
        horizontal = True,
    )
    threshold_value = threshold_label[1]

    # ---- Score button ----
    if st.button("Score order", type = "primary"):
        inputs = {
            "market": market,
            "order_region": order_region,
            "order_country": order_country,
            "order_state": order_state,
            "order_city": order_city,
            "customer_country": customer_country,
            "customer_state": customer_state,
            "shipping_mode": shipping_mode,
            "type": type_,
            "customer_segment": customer_segment,
            "category_name": category_name,
            "department_name": department_name,
            "product_name": product_name,
            "product_image": product_image,
            "days_for_shipment_scheduled": days_scheduled,
            "order_item_quantity": order_item_quantity,
            "sales_per_customer": sales_per_customer,
            "order_item_discount": order_item_discount,
            "order_item_discount_rate": order_item_discount_rate,
            "order_item_product_price": order_item_product_price,
            "product_price": product_price,
            "category_id": category_id,
            "department_id": department_id,
            "product_category_id": product_category_id,
            "product_status": product_status,
        }
        result = inference.predict(inputs)

        # Override classification with the user's threshold choice
        anchors = {0.90: 0.9981, 0.80: 0.8026, 0.70: 0.7160, 0.50: 0.4113}
        cutoff = anchors[threshold_value]
        flagged = result["probability"] >= cutoff

        # ---- Results panel ----
        st.markdown("---")
        st.subheader("Result")
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.metric("Predicted late probability", f"{result['probability']:.1%}")
            if result["above_base_rate"]:
                st.caption(f"Above base rate ({result['base_rate']:.1%})")
            else:
                st.caption(f"Below base rate ({result['base_rate']:.1%})")

        with rc2:
            pr = result["percentile_rank"]
            # pr = 0.90 means "riskier than 90% of test orders"
            # equivalently, "top 10% of predicted risk"
            top_pct = int(round((1 - pr) * 100))
            if top_pct <= 1:
                st.metric("Risk tier", "Top 1%")
                st.caption("Highest-priority intervention band")
            elif top_pct <= 5:
                st.metric("Risk tier", "Top 5%")
                st.caption("Second-highest intervention band")
            elif top_pct <= 10:
                st.metric("Risk tier", "Top 10%")
                st.caption("Precision-optimized threshold")
            elif top_pct <= 20:
                st.metric("Risk tier", "Top 20%")
                st.caption("Balanced precision band")
            elif top_pct <= 30:
                st.metric("Risk tier", "Top 30%")
                st.caption("Recommended operating threshold")
            else:
                st.metric("Risk tier", "Bottom 50%")
                st.caption("Below median risk")

        with rc3:
            label = "Flag" if flagged else "Standard"
            st.metric("Decision", label)
            st.caption(f"Threshold: top {int((1 - threshold_value) * 100)}%")

        # ---- Interpretation note ----
        # Use the same threshold the decision used, so the message and
        # decision are always coherent.
        if flagged:
            if result["probability"] >= 0.90:
                st.error(
                    "**High confidence late.** This order sits in the top tier "
                    "of predicted risk. Historically, orders like this are late "
                    "more than 9 times out of 10."
                )
            else:
                st.warning(
                    "**Flagged for intervention.** This order exceeds the "
                    "selected intervention threshold. If you have capacity to "
                    "review or expedite a subset of orders, this one is a "
                    "candidate."
                )
        else:
            st.success(
                "**Standard handling.** The model expects this order to arrive "
                "within its SLA at the selected intervention threshold. "
                "Calibration tests show predictions in this range are reliable "
                "to within ±3.5 percentage points."
            )


# ---------------------------------------------------------------------------
# TAB 2 - Business Context
# ---------------------------------------------------------------------------

with tab_context:
    st.markdown(
        "The model sits inside a wider SCOR framework. These are the "
        "enterprise-level numbers that motivate its use."
    )

    st.subheader("SCOR KPI baseline")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Late delivery rate", "54.9%", help = "Share of orders breaching their delivery SLA")
    k2.metric("Revenue at risk", "$15.9M", help = "Sales associated with late-flagged orders")
    k3.metric("Total net sales", "$29.1M")
    k4.metric("Avg profit margin", "12.0%")

    st.markdown("---")

    st.subheader("Profit leakage decomposition")
    st.markdown(
        """
        The supply chain generates value but does not retain it. Three leakage
        channels drain reported profit:

        | Channel | Magnitude | Share of reported profit |
        |---|---|---|
        | Gross Sales | $29.5M | - |
        | Reported Profit | $3.48M | 100% |
        | Discounts | -$3.26M | -93.7% |
        | Late delivery penalties | -$477K | -13.7% |
        | Returns and cancellations | -$73K | -2.1% |
        | **Adjusted profit** | **-$260K** | -7.5% |

        **Late delivery is the second-largest leakage channel.** Discount
        governance is the highest-impact lever overall, but it is a pricing
        policy decision. Late delivery is the channel that this model can
        address directly, at the per-order level.
        """
    )

    st.markdown("---")

    st.subheader("Shipping mode performance")
    st.markdown(
        """
        Notebook 6 established that late-delivery risk is driven by service-
        tier design, not logistics capability. The network delivers in a
        uniform 2-6 day window; what varies is the SLA promise attached to
        each mode.
        """
    )
    mode_df = pd.DataFrame({
        "Mode": ["Standard Class", "Second Class", "First Class", "Same Day"],
        "Orders": [39324, 12778, 10079, 3571],
        "Avg days": [3.99, 3.99, 2.00, 0.48],
        "Late rate": ["38.1%", "76.8%", "95.3%", "45.7%"],
    })
    st.dataframe(mode_df, hide_index = True, use_container_width = True)

    st.info(
        "**The First Class paradox.** Premium mode delivers with perfect "
        "consistency at 2 days, yet 95.3% of First Class shipments are "
        "classified late. The SLA promise is 1 day; the network delivers in "
        "2. The fix is recalibration, not a new carrier."
    )


# ---------------------------------------------------------------------------
# TAB 3 - Demand Planning Context
# ---------------------------------------------------------------------------

with tab_forecast:
    st.markdown(
        "Notebook 3 built a demand forecasting model for two priority segments. "
        "This tab shows the forecast output as read-only context - the "
        "forecasting work is not interactive here."
    )

    try:
        forecast_df = pd.read_parquet(ARTIFACT_DIR / "FactForecastWeekly.parquet")
    except FileNotFoundError:
        st.warning(
            "FactForecastWeekly.parquet not found in artifacts/. "
            "Copy it from the Notebook 3 output."
        )
        forecast_df = None

    if forecast_df is not None:
        segment = st.selectbox(
            "Segment",
            forecast_df["Market"].unique(),
            format_func = lambda m: f"{m} | Cleats",
        )
        subset = forecast_df[forecast_df["Market"] == segment].sort_values("Week_Start")

        chart_df = subset.set_index("Week_Start")[["ML_Forecast", "MA_Forecast"]]
        chart_df.columns = ["ML forecast (momentum)", "MA baseline (stability)"]
        st.line_chart(chart_df)

        st.caption(
            "The 12-week forecast pairs a momentum-aware ML projection with a "
            "variance-dampening moving-average baseline. The MA baseline was "
            "more accurate at aggregate grain; the ML model earned its place "
            "only in the Europe | Cleats segment after shock adjustment. Both "
            "are shown so the tradeoff is explicit."
        )


# ---------------------------------------------------------------------------
# TAB 4 - About the Model
# ---------------------------------------------------------------------------

with tab_about:
    metrics = inference.get_metrics()
    importances = inference.get_importances()

    st.subheader("Model card")

    st.markdown(f"""
    | Field | Value |
    |---|---|
    | Task | Binary classification - late vs on-time |
    | Target | `late_delivery_risk` (1 = late, 0 = on-time) |
    | Model | LightGBM (800 trees, lr 0.03) |
    | Training set | 127,810 orders |
    | Test set | 31,953 orders |
    | Base rate | {metrics['base_rate_test']:.1%} |
    | ROC-AUC | {metrics['lgbm']['roc_auc']:.4f} |
    | PR-AUC | {metrics['lgbm']['pr_auc']:.4f} |
    | Mean predicted | {metrics['calibration_mean_predicted']:.4f} |
    | Max decile calibration error | {metrics['calibration_max_bin_delta']:.4f} |
    | Data horizon | 2015-01 to 2018-01 |
    | Features | 25 (14 categorical, 11 numeric) |
    """)

    st.markdown("---")

    st.subheader("Feature importance - what drives the model")
    top_n = importances.head(15).copy()
    top_n.columns = ["Feature", "Importance"]
    top_n = top_n.set_index("Feature")
    st.bar_chart(top_n)

    st.caption(
        "Split-based importance. Order value, discount intensity, and payment "
        "channel dominate. Shipping mode enters as four separate one-hot "
        "levels - individually modest, collectively the second-largest group."
    )

    st.markdown("---")

    st.subheader("Excluded features - leakage audit")
    st.markdown("""
    An earlier version of this analysis leaked post-outcome information.
    The deployed model is trained on **pre-dispatch signals only**. The
    following 30 columns were excluded:

    | Category | Columns |
    |----------|---------|
    | Target | `late_delivery_risk` |
    | Post-outcome | `delivery_status`, `actual_delivery_date`, `shipping_delay`, `days_for_shipping_real`, `processing_days`, `order_status` |
    | Financial outcome | `benefit_per_order`, `profit_margin`, `order_profit_per_order`, `order_item_profit_ratio`, `order_item_total`, `sales` |
    | Identifiers and timestamps | `customer_id`, `order_id`, `order_item_id`, `product_card_id`, `order_item_cardprod_id`, `customer_email`, `customer_password`, `customer_fname`, `customer_lname`, `customer_street`, `customer_city`, `customer_zipcode`, `order_customer_id`, `order_date`, `shipping_date` |
    | Synthetic noise | `latitude`, `longitude` |

    The latitude/longitude exclusion is specific to this dataset: the values
    cluster in a single region off the coast of North Carolina regardless of
    the actual destination, making them noise rather than geography. See
    Notebook 5 for the full audit.
    """)

    st.markdown("---")

    st.subheader("Interpretation caveats")
    st.markdown("""
    - **Predictions are calibrated.** The mean predicted probability
      (0.5492) aligns with the observed late rate (0.5488). Decile-level
      error is within ±3.5 percentage points across the full range.
    - **The threshold is percentile-based, not fixed at 0.5.** The default
      operating point is the top 30% of predicted risk, where precision is
      0.886 and recall is 0.484. This can be shifted in the Live Scorer
      tab to see the tradeoff.
    - **Feature importance is not causation.** Shipping mode and payment
      type correlate with lateness because they reflect different handling
      workflows, not because changing them alone changes physics.
    - **Correlations are signals.** The protective effect of `type_TRANSFER`
      is consistent and strong, but the data cannot distinguish it from
      correlated unobserved factors. It should be treated as an investigative lead.
    """)

st.markdown("---")
st.markdown("""
<div style = "text-align: center; color: #666; padding: 20px;">
    <h3>DataCo Smart Supply Chain Analytics</h3>
    <p>Built with Streamlit | Framework: SCOR | Dataset: DataCo Smart Supply Chain for Big Data Analysis</p>
    <p>Late-Delivery Risk | Demand Forecasting | SCOR-Aligned KPIs</p>
    <hr>
    <p style = "font-size: 0.8em;">
        End-to-end supply chain analytics pipeline:
        Business Context → Data Engineering → EDA → Feature Engineering →
        Demand Forecasting → Risk Modeling → SCOR KPI Framework → Deployment.
    </p>
    <p style = "font-size: 0.75em; color: #999;">
        This app is the deployment layer of the project - live risk scorer, business context,
        demand planning outlook, and model card.
    </p>
</div>
""", unsafe_allow_html = True)