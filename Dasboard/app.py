# ============================================================
# OLIST E-COMMERCE ECOSYSTEM ANALYTICS
# ============================================================
#
# Production-Grade Streamlit Executive Dashboard
#
# Features
# --------
# 1. Single scrolling executive landing page
# 2. 9 dynamic KPI cards
# 3. Professional monthly LINE chart as first visualization
# 4. Revenue / order / customer trend analysis
# 5. Logistics & delivery analysis
# 6. Customer satisfaction analysis
# 7. Seller concentration / Pareto analysis
# 8. Category and payment analytics
# 9. Geographic Brazil analysis
# 10. Freight friction analysis
# 11. Customer retention funnel
# 12. Customer order-frequency distribution
# 13. Category performance matrix
# 14. Dynamic 2-page PDF executive report
# 15. PDF based on currently applied filters
# 16. Dark / light Streamlit compatibility
# 17. Cached data loading
# 18. Graceful NaN / date handling
#
# Expected CSV:
# ../Cleaned_Data/master_olist_cleaned_final.csv
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Olist Executive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

DATA_PATH = "../Cleaned_Data/master_olist_cleaned_final.csv"

APP_TITLE = "Olist E-Commerce Ecosystem Analytics"

REPORT_FILE_NAME = "Olist_Executive_Report.pdf"

REVIEW_BENCHMARK = 4.0

DELIVERY_WINDOWS = [
    "0-5 Days",
    "6-10 Days",
    "11-15 Days",
    "16-20 Days",
    "21-25 Days",
    "26-30 Days",
    "30+ Days",
]

FREIGHT_WINDOWS = [
    "<10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    ">50%",
]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "order_id",
    "customer_unique_id",
    "seller_id",
    "total_price",
    "total_freight",
    "review_score",
    "actual_delivery_time_days",
    "is_late",
    "primary_payment_type",
    "customer_state",
    "product_category_name_english",
    "order_purchase_timestamp",
]


# ============================================================
# BRAZIL STATE COORDINATES
# ============================================================

BRAZIL_STATE_COORDS = {
    "AC": (-8.77, -70.55),
    "AL": (-9.71, -35.73),
    "AP": (1.41, -51.77),
    "AM": (-3.47, -65.10),
    "BA": (-12.96, -41.70),
    "CE": (-5.20, -39.53),
    "DF": (-15.78, -47.93),
    "ES": (-19.19, -40.34),
    "GO": (-15.98, -49.86),
    "MA": (-5.42, -45.44),
    "MT": (-12.64, -55.42),
    "MS": (-20.51, -54.54),
    "MG": (-18.10, -44.38),
    "PA": (-3.79, -52.48),
    "PB": (-7.28, -36.72),
    "PR": (-24.89, -51.55),
    "PE": (-8.38, -37.86),
    "PI": (-7.72, -42.73),
    "RJ": (-22.25, -42.66),
    "RN": (-5.81, -36.59),
    "RS": (-30.17, -53.50),
    "RO": (-10.83, -63.34),
    "RR": (1.99, -61.33),
    "SC": (-27.45, -50.95),
    "SP": (-22.19, -48.79),
    "SE": (-10.57, -37.45),
    "TO": (-10.25, -48.25),
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.55rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.68;
        margin-bottom: 1.4rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 0.3rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        opacity: 0.68;
        margin-bottom: 1rem;
    }

    .insight-box {
        padding: 0.8rem;
        border-radius: 0.65rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.7rem;
    }

    .small-muted {
        opacity: 0.65;
        font-size: 0.85rem;
    }

    [data-testid="stMetric"] {
        padding: 0.75rem 0.35rem;
    }

    [data-testid="stMetricValue"] {
        font-weight: 750;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data(show_spinner="Loading Olist data...")
def load_data(path):

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    numeric_cols = [
        "total_price",
        "total_freight",
        "review_score",
        "actual_delivery_time_days",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    for col in [
        "order_id",
        "customer_unique_id",
        "seller_id",
    ]:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    for col in [
        "primary_payment_type",
        "customer_state",
        "product_category_name_english",
    ]:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        df[col] = df[col].replace(
            {
                "": "Unknown",
                "nan": "Unknown",
                "None": "Unknown",
            }
        )

    # --------------------------------------------------------
    # Boolean conversion
    # --------------------------------------------------------

    df["is_late"] = (
        df["is_late"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "late",
            ]
        )
    )

    # --------------------------------------------------------
    # Invalid values
    # --------------------------------------------------------

    df.loc[
        df["total_price"] < 0,
        "total_price",
    ] = np.nan

    df.loc[
        df["total_freight"] < 0,
        "total_freight",
    ] = np.nan

    df.loc[
        ~df["review_score"].between(1, 5),
        "review_score",
    ] = np.nan

    df.loc[
        df["actual_delivery_time_days"] < 0,
        "actual_delivery_time_days",
    ] = np.nan

    # --------------------------------------------------------
    # Freight ratio
    # --------------------------------------------------------

    df["freight_price_ratio"] = np.where(
        df["total_price"] > 0,
        df["total_freight"] / df["total_price"],
        np.nan,
    )

    # --------------------------------------------------------
    # Date helper
    # --------------------------------------------------------

    df["purchase_date"] = (
        df["order_purchase_timestamp"]
        .dt.date
    )

    df["purchase_month"] = (
        df["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data(DATA_PATH)

except FileNotFoundError:

    st.error(
        f"Data file not found:\n\n{DATA_PATH}"
    )

    st.stop()

except Exception as exc:

    st.error(
        f"Unable to load dataset: {exc}"
    )

    st.stop()


# ============================================================
# PLOTLY THEME
# ============================================================

def get_plotly_template():

    try:

        base = st.get_option(
            "theme.base"
        )

        if base == "dark":
            return "plotly_dark"

        if base == "light":
            return "plotly_white"

        return "plotly"

    except Exception:

        return "plotly"


PLOTLY_TEMPLATE = get_plotly_template()


# ============================================================
# PLOTLY STANDARDIZER
# ============================================================

def style_plot(fig, height=None):

    layout_kwargs = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=30,
            r=25,
            t=65,
            b=45,
        ),
        hoverlabel=dict(
            namelength=-1
        ),
    )

    if height is not None:
        layout_kwargs["height"] = height

    fig.update_layout(
        **layout_kwargs
    )

    return fig


# ============================================================
# FORMAT HELPERS
# ============================================================

def currency(value):

    if value is None or pd.isna(value):
        return "R$ 0.00"

    return f"R$ {value:,.2f}"


def compact_currency(value):

    if value is None or pd.isna(value):
        return "R$ 0"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"R$ {value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"R$ {value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"R$ {value / 1_000:.1f}K"

    return f"R$ {value:,.0f}"


def number(value):

    if value is None or pd.isna(value):
        return "0"

    return f"{int(round(value)):,}"


def percent(value):

    if value is None or pd.isna(value):
        return "0.0%"

    return f"{value:.1f}%"


def score(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2f} ★"


def safe_mean(series):

    clean = series.dropna()

    if clean.empty:
        return np.nan

    return clean.mean()


def title_case(value):

    if value is None:
        return "Unknown"

    return (
        str(value)
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Executive Filters")

st.sidebar.caption(
    "Every KPI, visualization and PDF recommendation "
    "updates from these filters."
)

valid_dates = (
    df["order_purchase_timestamp"]
    .dropna()
)

if valid_dates.empty:

    st.error(
        "No valid purchase dates are available."
    )

    st.stop()


min_date = valid_dates.min().date()
max_date = valid_dates.max().date()


date_range = st.sidebar.date_input(
    "Order Purchase Date",
    value=(
        min_date,
        max_date,
    ),
    min_value=min_date,
    max_value=max_date,
)


if isinstance(
    date_range,
    (tuple, list)
):

    if len(date_range) == 2:

        start_date = date_range[0]
        end_date = date_range[1]

    else:

        start_date = date_range[0]
        end_date = date_range[0]

else:

    start_date = date_range
    end_date = date_range


states = sorted(
    df["customer_state"]
    .dropna()
    .unique()
)


selected_state = st.sidebar.selectbox(
    "Customer State",
    ["All States"] + states,
)


payment_types = sorted(
    df["primary_payment_type"]
    .dropna()
    .unique()
)


selected_payment_types = st.sidebar.multiselect(
    "Payment Type",
    payment_types,
    default=payment_types,
)


st.sidebar.divider()

if st.sidebar.button(
    "🔄 Clear Cache & Reload",
    use_container_width=True,
    key="reload_data_button",
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["purchase_date"].between(
        start_date,
        end_date,
    )
]


if selected_state != "All States":

    filtered_df = filtered_df[
        filtered_df["customer_state"]
        == selected_state
    ]


if selected_payment_types:

    filtered_df = filtered_df[
        filtered_df[
            "primary_payment_type"
        ].isin(
            selected_payment_types
        )
    ]

else:

    filtered_df = filtered_df.iloc[0:0]


# ============================================================
# EMPTY DATA
# ============================================================

if filtered_df.empty:

    st.warning(
        "No records match the selected filters. "
        "Please broaden the date range or filter selection."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="main-title">
        📊 {APP_TITLE}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Executive command center for revenue, customers,
        logistics, satisfaction, freight and seller concentration.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Period: {start_date.strftime('%d %b %Y')} → "
    f"{end_date.strftime('%d %b %Y')}  |  "
    f"Filtered records: {len(filtered_df):,}"
)


# ============================================================
# CORE KPI CALCULATIONS
# ============================================================

total_gmv = filtered_df[
    "total_price"
].sum()

total_orders = filtered_df[
    "order_id"
].nunique()

total_customers = filtered_df[
    "customer_unique_id"
].nunique()

total_sellers = filtered_df[
    "seller_id"
].nunique()

average_order_value = (
    total_gmv / total_orders
    if total_orders > 0
    else np.nan
)

average_review = safe_mean(
    filtered_df["review_score"]
)

average_delivery = safe_mean(
    filtered_df[
        "actual_delivery_time_days"
    ]
)

late_delivery_rate = (
    filtered_df["is_late"].mean() * 100
)

total_freight = filtered_df[
    "total_freight"
].sum()

freight_ratio = (
    total_freight / total_gmv * 100
    if total_gmv > 0
    else np.nan
)

customer_orders = (
    filtered_df
    .groupby(
        "customer_unique_id"
    )["order_id"]
    .nunique()
)

repeat_customer_count = int(
    (customer_orders > 1).sum()
)

repeat_buyer_rate = (
    repeat_customer_count
    / total_customers
    * 100
    if total_customers > 0
    else 0
)

one_time_customers = int(
    (customer_orders == 1).sum()
)

two_order_customers = int(
    (customer_orders == 2).sum()
)

three_plus_customers = int(
    (customer_orders >= 3).sum()
)


# ============================================================
# EXECUTIVE KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Commercial, customer and operational health at a glance."
    "</div>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# KPI ROW 1
# ------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)

with k1:

    st.metric(
        "Total GMV",
        compact_currency(total_gmv),
    )

with k2:

    st.metric(
        "Total Orders",
        number(total_orders),
    )

with k3:

    st.metric(
        "Average Review",
        score(average_review),
    )

with k4:

    st.metric(
        "Avg Delivery",
        (
            f"{average_delivery:.1f} days"
            if not pd.isna(average_delivery)
            else "N/A"
        ),
    )

with k5:

    st.metric(
        "Repeat Buyer Rate",
        percent(repeat_buyer_rate),
    )


# ------------------------------------------------------------
# KPI ROW 2
# ------------------------------------------------------------

k6, k7, k8, k9 = st.columns(4)

with k6:

    st.metric(
        "Customers",
        number(total_customers),
    )

with k7:

    st.metric(
        "Active Sellers",
        number(total_sellers),
    )

with k8:

    st.metric(
        "Average Order Value",
        currency(average_order_value),
    )

with k9:

    st.metric(
        "Late Delivery Rate",
        percent(late_delivery_rate),
    )


# ============================================================
# SECTION 1
# MONTHLY PERFORMANCE
# LINE CHART MUST BE FIRST
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📈 01 — Revenue & Business Momentum"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "The primary trend view tracks how GMV, orders and customer activity evolve over time."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MONTHLY AGGREGATION
# ============================================================

monthly = (
    filtered_df
    .groupby("purchase_month")
    .agg(
        GMV=("total_price", "sum"),
        Orders=("order_id", "nunique"),
        Customers=(
            "customer_unique_id",
            "nunique",
        ),
        Average_Review=(
            "review_score",
            "mean",
        ),
    )
    .reset_index()
)

monthly["purchase_month"] = pd.to_datetime(
    monthly["purchase_month"]
)

monthly = monthly.sort_values(
    "purchase_month"
)


# ============================================================
# CHART 1 — PROFESSIONAL LINE CHART
# ============================================================

fig_line = go.Figure()

fig_line.add_trace(
    go.Scatter(
        x=monthly["purchase_month"],
        y=monthly["GMV"],
        mode="lines+markers",
        name="GMV",
        line=dict(
            width=3,
            shape="spline",
        ),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "GMV: R$ %{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_line.update_layout(
    title="Monthly GMV Momentum",
    xaxis_title="Month",
    yaxis_title="GMV (R$)",
    hovermode="x unified",
)

fig_line = style_plot(
    fig_line,
    height=450,
)

st.plotly_chart(
    fig_line,
    use_container_width=True,
)


# ============================================================
# SECONDARY TREND CHARTS
# ============================================================

trend1, trend2 = st.columns(2)


# ============================================================
# CHART 2 — MONTHLY ORDERS
# ============================================================

with trend1:

    fig_orders = px.area(
        monthly,
        x="purchase_month",
        y="Orders",
        markers=True,
        title="Monthly Order Volume",
        labels={
            "purchase_month": "Month",
            "Orders": "Orders",
        },
    )

    fig_orders = style_plot(
        fig_orders,
        height=370,
    )

    st.plotly_chart(
        fig_orders,
        use_container_width=True,
    )


# ============================================================
# CHART 3 — MONTHLY REVIEW
# ============================================================

with trend2:

    fig_reviews = go.Figure()

    fig_reviews.add_trace(
        go.Scatter(
            x=monthly["purchase_month"],
            y=monthly["Average_Review"],
            mode="lines+markers",
            name="Average Review",
            line=dict(
                width=3,
                shape="spline",
            ),
        )
    )

    fig_reviews.add_hline(
        y=REVIEW_BENCHMARK,
        line_dash="dash",
        annotation_text="4.0★ Benchmark",
    )

    fig_reviews.update_layout(
        title="Monthly Customer Satisfaction",
        xaxis_title="Month",
        yaxis_title="Average Review",
        yaxis=dict(
            range=[0, 5.2]
        ),
    )

    fig_reviews = style_plot(
        fig_reviews,
        height=370,
    )

    st.plotly_chart(
        fig_reviews,
        use_container_width=True,
    )


# ============================================================
# SECTION 2
# LOGISTICS & SATISFACTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🚚 02 — Logistics & Customer Satisfaction"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Identify the relationship between delivery speed, lateness and customer ratings."
    "</div>",
    unsafe_allow_html=True,
)


logistics1, logistics2 = st.columns(2)


# ============================================================
# CHART 4 — DELIVERY SPEED VS REVIEW
# ============================================================

delivery_data = filtered_df[
    filtered_df[
        "actual_delivery_time_days"
    ].notna()
    &
    filtered_df[
        "review_score"
    ].notna()
].copy()

if not delivery_data.empty:

    delivery_bins = [
        -np.inf,
        5,
        10,
        15,
        20,
        25,
        30,
        np.inf,
    ]

    delivery_data[
        "delivery_window"
    ] = pd.cut(
        delivery_data[
            "actual_delivery_time_days"
        ],
        bins=delivery_bins,
        labels=DELIVERY_WINDOWS,
        include_lowest=True,
    )

    delivery_rating = (
        delivery_data
        .groupby(
            "delivery_window",
            observed=False,
        )["review_score"]
        .mean()
        .reset_index()
    )

    with logistics1:

        fig_delivery = px.bar(
            delivery_rating,
            x="delivery_window",
            y="review_score",
            text="review_score",
            title="Rating Deterioration Across Delivery Speed",
            labels={
                "delivery_window": "Delivery Window",
                "review_score": "Average Review",
            },
        )

        fig_delivery.update_traces(
            texttemplate="%{text:.2f}★",
            textposition="outside",
        )

        fig_delivery.add_hline(
            y=4,
            line_dash="dash",
            annotation_text="4.0★ Benchmark",
        )

        fig_delivery.update_yaxes(
            range=[0, 5.2]
        )

        fig_delivery = style_plot(
            fig_delivery,
            height=420,
        )

        st.plotly_chart(
            fig_delivery,
            use_container_width=True,
        )


# ============================================================
# CHART 5 — ON-TIME VS LATE
# ============================================================

with logistics2:

    comparison = filtered_df[
        filtered_df["review_score"].notna()
    ].copy()

    comparison["Delivery Status"] = np.where(
        comparison["is_late"],
        "Late",
        "On-Time",
    )

    status_review = (
        comparison
        .groupby(
            "Delivery Status"
        )["review_score"]
        .mean()
        .reindex(
            ["On-Time", "Late"]
        )
        .reset_index()
    )

    fig_status = px.bar(
        status_review,
        x="Delivery Status",
        y="review_score",
        color="Delivery Status",
        text="review_score",
        title="Customer Rating: On-Time vs Late",
        labels={
            "review_score": "Average Review",
        },
    )

    fig_status.update_traces(
        texttemplate="%{text:.2f}★",
        textposition="outside",
    )

    fig_status.add_hline(
        y=4,
        line_dash="dash",
        annotation_text="4.0★ Benchmark",
    )

    fig_status.update_yaxes(
        range=[0, 5.2]
    )

    fig_status = style_plot(
        fig_status,
        height=420,
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True,
    )


# ============================================================
# CHART 6 — DELIVERY DISTRIBUTION
# ============================================================

fig_delivery_hist = px.histogram(
    delivery_data,
    x="actual_delivery_time_days",
    nbins=30,
    title="Distribution of Actual Delivery Time",
    labels={
        "actual_delivery_time_days":
            "Delivery Time (Days)",
        "count":
            "Orders",
    },
)

fig_delivery_hist.add_vline(
    x=average_delivery,
    line_dash="dash",
    annotation_text=(
        f"Mean: {average_delivery:.1f} days"
    ),
)

fig_delivery_hist = style_plot(
    fig_delivery_hist,
    height=380,
)

st.plotly_chart(
    fig_delivery_hist,
    use_container_width=True,
)


# ============================================================
# SECTION 3
# REVENUE CONCENTRATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "💰 03 — Revenue Concentration & Seller Economics"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Measure seller dependency and identify concentration risk within the marketplace."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SELLER REVENUE
# ============================================================

seller_revenue = (
    filtered_df
    .groupby("seller_id")[
        "total_price"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

seller_revenue = seller_revenue[
    seller_revenue > 0
]


# ============================================================
# PARETO
# ============================================================

if not seller_revenue.empty:

    pareto = pd.DataFrame(
        {
            "seller_id":
                seller_revenue.index,
            "revenue":
                seller_revenue.values,
        }
    )

    pareto[
        "cumulative_revenue_pct"
    ] = (
        pareto["revenue"].cumsum()
        / pareto["revenue"].sum()
        * 100
    )

    pareto[
        "seller_pct"
    ] = (
        np.arange(
            1,
            len(pareto) + 1,
        )
        / len(pareto)
        * 100
    )

    actual_80 = (
        pareto[
            "cumulative_revenue_pct"
        ] >= 80
    )

    if actual_80.any():

        first_80_idx = (
            actual_80.idxmax()
        )

        seller_pct_80 = pareto.loc[
            first_80_idx,
            "seller_pct",
        ]

    else:

        seller_pct_80 = 100


    pareto_col1, pareto_col2 = st.columns(2)


    # --------------------------------------------------------
    # CHART 7 — PARETO
    # --------------------------------------------------------

    with pareto_col1:

        fig_pareto = go.Figure()

        fig_pareto.add_trace(
            go.Scatter(
                x=pareto["seller_pct"],
                y=pareto[
                    "cumulative_revenue_pct"
                ],
                mode="lines",
                name="Cumulative GMV",
                line=dict(
                    width=3,
                    shape="spline",
                ),
                hovertemplate=(
                    "Sellers: %{x:.1f}%<br>"
                    "Cumulative GMV: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig_pareto.add_vline(
            x=20,
            line_dash="dash",
            annotation_text="20% Sellers",
        )

        fig_pareto.add_hline(
            y=80,
            line_dash="dash",
            annotation_text="80% GMV",
        )

        fig_pareto.add_vline(
            x=seller_pct_80,
            line_dash="dot",
            annotation_text=(
                f"Actual {seller_pct_80:.1f}%"
            ),
        )

        fig_pareto.update_layout(
            title="Seller Pareto / Lorenz Curve",
            xaxis_title="% of Active Sellers",
            yaxis_title="Cumulative GMV %",
            hovermode="x unified",
        )

        fig_pareto = style_plot(
            fig_pareto,
            height=430,
        )

        st.plotly_chart(
            fig_pareto,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # CHART 8 — TOP SELLERS
    # --------------------------------------------------------

    with pareto_col2:

        top_sellers = (
            seller_revenue
            .head(15)
            .sort_values()
            .reset_index()
        )

        top_sellers.columns = [
            "seller_id",
            "GMV",
        ]

        fig_sellers = px.bar(
            top_sellers,
            x="GMV",
            y="seller_id",
            orientation="h",
            title="Top 15 Sellers by GMV",
            text="GMV",
            labels={
                "GMV": "GMV (R$)",
                "seller_id": "Seller",
            },
        )

        fig_sellers.update_traces(
            texttemplate="R$ %{text:,.0f}",
            textposition="outside",
        )

        fig_sellers = style_plot(
            fig_sellers,
            height=430,
        )

        st.plotly_chart(
            fig_sellers,
            use_container_width=True,
        )


# ============================================================
# SELLER KPI STRIP
# ============================================================

top_1_share = np.nan
top_5_share = np.nan
top_10_share = np.nan

if not seller_revenue.empty:

    total_seller_gmv = seller_revenue.sum()

    top_1_share = (
        seller_revenue.head(1).sum()
        / total_seller_gmv
        * 100
    )

    top_5_share = (
        seller_revenue.head(5).sum()
        / total_seller_gmv
        * 100
    )

    top_10_share = (
        seller_revenue.head(10).sum()
        / total_seller_gmv
        * 100
    )


s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Top 1 Seller Share",
        percent(top_1_share),
    )

with s2:
    st.metric(
        "Top 5 Seller Share",
        percent(top_5_share),
    )

with s3:
    st.metric(
        "Top 10 Seller Share",
        percent(top_10_share),
    )

with s4:
    st.metric(
        "80% GMV Threshold",
        percent(seller_pct_80),
    )


# ============================================================
# SECTION 4
# CATEGORY & PAYMENT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🛍️ 04 — Category, Payment & Revenue Mix"
    "</div>",
    unsafe_allow_html=True,
)

category_col1, category_col2 = st.columns(2)


# ============================================================
# CHART 9 — TOP CATEGORIES GMV
# ============================================================

category_revenue = (
    filtered_df
    .groupby(
        "product_category_name_english"
    )["total_price"]
    .sum()
    .sort_values(
        ascending=False
    )
)

top_category_df = (
    category_revenue
    .head(15)
    .sort_values()
    .reset_index()
)

top_category_df.columns = [
    "Category",
    "GMV",
]

with category_col1:

    fig_categories = px.bar(
        top_category_df,
        x="GMV",
        y="Category",
        orientation="h",
        title="Top 15 Product Categories by GMV",
        text="GMV",
    )

    fig_categories.update_traces(
        texttemplate="R$ %{text:,.0f}",
        textposition="outside",
    )

    fig_categories = style_plot(
        fig_categories,
        height=500,
    )

    st.plotly_chart(
        fig_categories,
        use_container_width=True,
    )


# ============================================================
# CHART 10 — PAYMENT REVENUE
# ============================================================

payment_revenue = (
    filtered_df
    .groupby(
        "primary_payment_type"
    )["total_price"]
    .sum()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

with category_col2:

    fig_payment = px.bar(
        payment_revenue,
        x="primary_payment_type",
        y="total_price",
        text="total_price",
        title="GMV by Primary Payment Type",
        labels={
            "primary_payment_type":
                "Payment Type",
            "total_price":
                "GMV (R$)",
        },
    )

    fig_payment.update_traces(
        texttemplate="R$ %{text:,.0f}",
        textposition="outside",
    )

    fig_payment = style_plot(
        fig_payment,
        height=500,
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True,
    )


# ============================================================
# CHART 11 — PAYMENT ORDER SHARE
# ============================================================

payment_orders = (
    filtered_df
    .groupby(
        "primary_payment_type"
    )["order_id"]
    .nunique()
    .reset_index(
        name="Orders"
    )
)

fig_payment_orders = px.pie(
    payment_orders,
    names="primary_payment_type",
    values="Orders",
    hole=0.55,
    title="Order Share by Payment Type",
)

fig_payment_orders = style_plot(
    fig_payment_orders,
    height=430,
)

st.plotly_chart(
    fig_payment_orders,
    use_container_width=True,
)


# ============================================================
# CHART 12 — SUNBURST
# ============================================================

category_payment = filtered_df[
    filtered_df["total_price"].notna()
].copy()

top_10_categories = (
    category_payment
    .groupby(
        "product_category_name_english"
    )["total_price"]
    .sum()
    .nlargest(10)
    .index
)

sunburst_data = category_payment[
    category_payment[
        "product_category_name_english"
    ].isin(
        top_10_categories
    )
]

sunburst_data = (
    sunburst_data
    .groupby(
        [
            "product_category_name_english",
            "primary_payment_type",
        ],
        as_index=False,
    )["total_price"]
    .sum()
)

fig_sunburst = px.sunburst(
    sunburst_data,
    path=[
        "product_category_name_english",
        "primary_payment_type",
    ],
    values="total_price",
    title=(
        "Revenue Hierarchy — "
        "Top 10 Categories × Payment Type"
    ),
)

fig_sunburst = style_plot(
    fig_sunburst,
    height=650,
)

st.plotly_chart(
    fig_sunburst,
    use_container_width=True,
)


# ============================================================
# SECTION 5
# GEOGRAPHIC ANALYTICS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🌎 05 — Geographic Performance"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Understand where customers, orders and GMV are concentrated across Brazil."
    "</div>",
    unsafe_allow_html=True,
)


geo = (
    filtered_df
    .groupby("customer_state")
    .agg(
        Orders=(
            "order_id",
            "nunique",
        ),
        GMV=(
            "total_price",
            "sum",
        ),
        Customers=(
            "customer_unique_id",
            "nunique",
        ),
        Avg_Review=(
            "review_score",
            "mean",
        ),
    )
    .reset_index()
)

geo["lat"] = geo[
    "customer_state"
].map(
    lambda x: BRAZIL_STATE_COORDS.get(
        str(x).upper(),
        (np.nan, np.nan),
    )[0]
)

geo["lon"] = geo[
    "customer_state"
].map(
    lambda x: BRAZIL_STATE_COORDS.get(
        str(x).upper(),
        (np.nan, np.nan),
    )[1]
)

geo = geo.dropna(
    subset=[
        "lat",
        "lon",
    ]
)


# ============================================================
# CHART 13 — BRAZIL GEO
# ============================================================

if not geo.empty:

    fig_geo = px.scatter_geo(
        geo,
        lat="lat",
        lon="lon",
        size="Orders",
        color="GMV",
        hover_name="customer_state",
        hover_data={
            "Orders": ":,",
            "Customers": ":,",
            "GMV": ":,.2f",
            "Avg_Review": ":.2f",
            "lat": False,
            "lon": False,
        },
        scope="south america",
        title=(
            "Customer Order & GMV Concentration Across Brazil"
        ),
    )

    fig_geo.update_geos(
        showcountries=True,
        showland=True,
        showocean=True,
        fitbounds="locations",
    )

    fig_geo = style_plot(
        fig_geo,
        height=600,
    )

    st.plotly_chart(
        fig_geo,
        use_container_width=True,
    )


# ============================================================
# CHART 14 — STATE GMV
# ============================================================

state_gmv = (
    filtered_df
    .groupby("customer_state")[
        "total_price"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(15)
    .sort_values()
    .reset_index()
)

state_gmv.columns = [
    "State",
    "GMV",
]

fig_state_gmv = px.bar(
    state_gmv,
    x="GMV",
    y="State",
    orientation="h",
    title="Top 15 Customer States by GMV",
    text="GMV",
)

fig_state_gmv.update_traces(
    texttemplate="R$ %{text:,.0f}",
    textposition="outside",
)

fig_state_gmv = style_plot(
    fig_state_gmv,
    height=500,
)

st.plotly_chart(
    fig_state_gmv,
    use_container_width=True,
)


# ============================================================
# SECTION 6
# FREIGHT FRICTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📦 06 — Freight Friction Analytics"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Analyze how freight cost relative to product price influences customer satisfaction."
    "</div>",
    unsafe_allow_html=True,
)


freight_data = filtered_df[
    filtered_df[
        "freight_price_ratio"
    ].notna()
    &
    filtered_df[
        "review_score"
    ].notna()
].copy()

if not freight_data.empty:

    freight_bins = [
        -np.inf,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        np.inf,
    ]

    freight_data[
        "freight_window"
    ] = pd.cut(
        freight_data[
            "freight_price_ratio"
        ],
        bins=freight_bins,
        labels=FREIGHT_WINDOWS,
        include_lowest=True,
    )

    freight_rating = (
        freight_data
        .groupby(
            "freight_window",
            observed=False,
        )
        .agg(
            Average_Review=(
                "review_score",
                "mean",
            ),
            Orders=(
                "order_id",
                "nunique",
            ),
        )
        .reset_index()
    )


    freight_col1, freight_col2 = st.columns(2)


    # --------------------------------------------------------
    # CHART 15
    # --------------------------------------------------------

    with freight_col1:

        fig_freight = px.bar(
            freight_rating,
            x="freight_window",
            y="Average_Review",
            text="Average_Review",
            title=(
                "Rating Impact by Freight-to-Price Ratio"
            ),
        )

        fig_freight.update_traces(
            texttemplate="%{text:.2f}★",
            textposition="outside",
        )

        fig_freight.add_hline(
            y=4,
            line_dash="dash",
            annotation_text="4.0★ Benchmark",
        )

        fig_freight.update_yaxes(
            range=[0, 5.2]
        )

        fig_freight = style_plot(
            fig_freight,
            height=430,
        )

        st.plotly_chart(
            fig_freight,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # CHART 16
    # --------------------------------------------------------

    with freight_col2:

        fig_freight_orders = px.bar(
            freight_rating,
            x="freight_window",
            y="Orders",
            text="Orders",
            title=(
                "Order Volume Across Freight Friction Bands"
            ),
        )

        fig_freight_orders.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
        )

        fig_freight_orders = style_plot(
            fig_freight_orders,
            height=430,
        )

        st.plotly_chart(
            fig_freight_orders,
            use_container_width=True,
        )


# ============================================================
# SECTION 7
# CUSTOMER RETENTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🔁 07 — Customer Retention & Loyalty"
    "</div>",
    unsafe_allow_html=True,
)

retention_col1, retention_col2 = st.columns(2)


# ============================================================
# CHART 17 — RETENTION FUNNEL
# ============================================================

funnel_df = pd.DataFrame(
    {
        "Segment": [
            "One-Time Buyers",
            "2 Orders",
            "3+ Orders",
        ],
        "Customers": [
            one_time_customers,
            two_order_customers,
            three_plus_customers,
        ],
    }
)

with retention_col1:

    fig_funnel = go.Figure(
        go.Funnel(
            y=funnel_df["Segment"],
            x=funnel_df["Customers"],
            textinfo="value+percent initial",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Customers: %{x:,}<br>"
                "Share: %{percentInitial:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig_funnel.update_layout(
        title="Customer Retention Funnel"
    )

    fig_funnel = style_plot(
        fig_funnel,
        height=470,
    )

    st.plotly_chart(
        fig_funnel,
        use_container_width=True,
    )


# ============================================================
# CHART 18 — ORDER FREQUENCY
# ============================================================

with retention_col2:

    frequency = (
        customer_orders
        .value_counts()
        .sort_index()
        .head(10)
        .reset_index()
    )

    frequency.columns = [
        "Orders_Per_Customer",
        "Customers",
    ]

    fig_frequency = px.bar(
        frequency,
        x="Orders_Per_Customer",
        y="Customers",
        text="Customers",
        title="Customer Order-Frequency Distribution",
        labels={
            "Orders_Per_Customer":
                "Orders per Customer",
            "Customers":
                "Customers",
        },
    )

    fig_frequency.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig_frequency = style_plot(
        fig_frequency,
        height=470,
    )

    st.plotly_chart(
        fig_frequency,
        use_container_width=True,
    )


# ============================================================
# RETENTION KPI
# ============================================================

r1, r2, r3, r4 = st.columns(4)

with r1:

    st.metric(
        "One-Time Customers",
        number(one_time_customers),
    )

with r2:

    st.metric(
        "2-Order Customers",
        number(two_order_customers),
    )

with r3:

    st.metric(
        "3+ Order Customers",
        number(three_plus_customers),
    )

with r4:

    st.metric(
        "Repeat Customers",
        number(repeat_customer_count),
    )


# ============================================================
# SECTION 8
# CATEGORY PERFORMANCE MATRIX
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📊 08 — Category Performance Matrix"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Compare category revenue against customer satisfaction."
    "</div>",
    unsafe_allow_html=True,
)


category_matrix = (
    filtered_df
    .groupby(
        "product_category_name_english"
    )
    .agg(
        GMV=("total_price", "sum"),
        Orders=("order_id", "nunique"),
        Avg_Review=("review_score", "mean"),
        Avg_Freight=("total_freight", "mean"),
    )
    .reset_index()
)

category_matrix = category_matrix[
    category_matrix["GMV"] > 0
]

category_matrix = category_matrix.nlargest(
    25,
    "GMV",
)


# ============================================================
# CHART 19 — CATEGORY SCATTER
# ============================================================

if not category_matrix.empty:

    fig_matrix = px.scatter(
        category_matrix,
        x="GMV",
        y="Avg_Review",
        size="Orders",
        color="Avg_Freight",
        hover_name=(
            "product_category_name_english"
        ),
        hover_data={
            "GMV": ":,.2f",
            "Orders": ":,",
            "Avg_Review": ":.2f",
            "Avg_Freight": ":,.2f",
        },
        title=(
            "Category Economics — GMV vs Customer Satisfaction"
        ),
        labels={
            "GMV": "GMV (R$)",
            "Avg_Review": "Average Review",
            "Avg_Freight": "Average Freight",
        },
    )

    fig_matrix.add_hline(
        y=4,
        line_dash="dash",
        annotation_text="4.0★ Benchmark",
    )

    fig_matrix = style_plot(
        fig_matrix,
        height=600,
    )

    st.plotly_chart(
        fig_matrix,
        use_container_width=True,
    )


# ============================================================
# CATEGORY TABLE
# ============================================================

display_category = category_matrix.copy()

display_category[
    "Category"
] = display_category[
    "product_category_name_english"
].apply(title_case)

display_category[
    "GMV"
] = display_category[
    "GMV"
].apply(currency)

display_category[
    "Avg Review"
] = display_category[
    "Avg_Review"
].round(2)

display_category[
    "Orders"
] = display_category[
    "Orders"
].astype(int)

display_category = display_category[
    [
        "Category",
        "GMV",
        "Orders",
        "Avg Review",
    ]
]

st.dataframe(
    display_category,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SECTION 9
# EXECUTIVE INSIGHTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "🧠 09 — Executive Intelligence"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# TOP STATE
# ============================================================

state_orders = (
    filtered_df
    .groupby("customer_state")[
        "order_id"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
)

if not state_orders.empty:

    top_state = state_orders.index[0]
    top_state_orders = state_orders.iloc[0]

else:

    top_state = "N/A"
    top_state_orders = 0


# ============================================================
# TOP CATEGORY
# ============================================================

if not category_revenue.empty:

    top_category = (
        category_revenue.index[0]
    )

    top_category_gmv = (
        category_revenue.iloc[0]
    )

else:

    top_category = "N/A"
    top_category_gmv = 0


# ============================================================
# BEST / WORST DELIVERY WINDOW
# ============================================================

if not delivery_data.empty:

    delivery_rating_sorted = (
        delivery_rating
        .dropna(
            subset=["review_score"]
        )
    )

    if not delivery_rating_sorted.empty:

        best_delivery_window = (
            delivery_rating_sorted
            .loc[
                delivery_rating_sorted[
                    "review_score"
                ].idxmax(),
                "delivery_window",
            ]
        )

        worst_delivery_window = (
            delivery_rating_sorted
            .loc[
                delivery_rating_sorted[
                    "review_score"
                ].idxmin(),
                "delivery_window",
            ]
        )

    else:

        best_delivery_window = "N/A"
        worst_delivery_window = "N/A"

else:

    best_delivery_window = "N/A"
    worst_delivery_window = "N/A"


# ============================================================
# INSIGHT CARDS
# ============================================================

i1, i2, i3 = st.columns(3)

with i1:

    st.markdown(
        f"""
        <div class="insight-box">
            <b>🏆 Leading Market</b><br>
            {top_state}<br>
            <span class="small-muted">
            {top_state_orders:,} orders
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with i2:

    st.markdown(
        f"""
        <div class="insight-box">
            <b>🛍️ Leading Category</b><br>
            {title_case(top_category)}<br>
            <span class="small-muted">
            {currency(top_category_gmv)} GMV
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with i3:

    st.markdown(
        f"""
        <div class="insight-box">
            <b>🚚 Delivery Risk Window</b><br>
            {worst_delivery_window}<br>
            <span class="small-muted">
            Lowest average rating among delivery windows
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AUTOMATED BUSINESS RECOMMENDATIONS
# ============================================================

recommendations = []


# Delivery recommendation
if late_delivery_rate >= 20:

    recommendations.append(
        "Delivery performance requires immediate attention. "
        f"The filtered late-delivery rate is {late_delivery_rate:.1f}%. "
        "Prioritize seller dispatch SLAs, fulfillment bottlenecks "
        "and high-volume logistics corridors."
    )

elif late_delivery_rate >= 10:

    recommendations.append(
        "Delivery reliability presents a moderate operational risk. "
        f"Late deliveries account for {late_delivery_rate:.1f}% of orders. "
        "Target the highest-volume delayed routes and sellers."
    )

else:

    recommendations.append(
        "Delivery reliability is relatively healthy. "
        f"The late-delivery rate is {late_delivery_rate:.1f}%. "
        "Continue monitoring service-level consistency."
    )


# Review recommendation
if not pd.isna(average_review):

    if average_review < 4:

        recommendations.append(
            f"Customer satisfaction is below the 4.0★ benchmark "
            f"at {average_review:.2f}★. "
            "Investigate delivery experience, freight friction "
            "and category-level service quality."
        )

    elif average_review < 4.5:

        recommendations.append(
            f"Customer satisfaction is acceptable at "
            f"{average_review:.2f}★ but leaves room for improvement. "
            "Focus on the lowest-rated categories and delivery windows."
        )

    else:

        recommendations.append(
            f"Customer satisfaction is strong at "
            f"{average_review:.2f}★. "
            "Protect high-performing service processes."
        )


# Retention recommendation
if repeat_buyer_rate < 20:

    recommendations.append(
        f"Retention is a strategic opportunity. "
        f"Only {repeat_buyer_rate:.1f}% of customers are repeat buyers. "
        "Consider personalized recommendations, lifecycle campaigns "
        "and post-purchase engagement."
    )

elif repeat_buyer_rate < 35:

    recommendations.append(
        f"Repeat purchasing is moderate at "
        f"{repeat_buyer_rate:.1f}%. "
        "Target one-time buyers with relevant reactivation campaigns."
    )

else:

    recommendations.append(
        f"Customer retention is comparatively strong at "
        f"{repeat_buyer_rate:.1f}%. "
        "Protect repeat customers with loyalty and personalization initiatives."
    )


# Seller concentration
if not pd.isna(top_5_share):

    if top_5_share >= 50:

        recommendations.append(
            f"Seller dependency is high: the top five sellers "
            f"represent approximately {top_5_share:.1f}% of GMV. "
            "Diversification and seller dependency monitoring are recommended."
        )

    else:

        recommendations.append(
            f"Seller concentration is more diversified, with the top "
            f"five sellers contributing approximately {top_5_share:.1f}% of GMV."
        )


# Freight
if not pd.isna(freight_ratio):

    if freight_ratio >= 20:

        recommendations.append(
            f"Freight represents approximately {freight_ratio:.1f}% "
            "of GMV. Review shipping economics, pricing strategy "
            "and high-friction destinations."
        )

    else:

        recommendations.append(
            f"Freight represents approximately {freight_ratio:.1f}% "
            "of GMV, suggesting comparatively controlled freight exposure."
        )


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

for idx, recommendation in enumerate(
    recommendations,
    start=1,
):

    st.markdown(
        f"""
        <div class="insight-box">
            <b>{idx}. Recommendation</b><br>
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def build_pdf_report(
    data,
    start_date,
    end_date,
    state,
    payments,
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title="Olist Executive Analytics Report",
        author="Olist Analytics Dashboard",
    )


    # --------------------------------------------------------
    # PDF STYLES
    # --------------------------------------------------------

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "ExecutiveTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=5,
    )


    subtitle_style = ParagraphStyle(
        "ExecutiveSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        spaceAfter=10,
    )


    heading_style = ParagraphStyle(
        "ExecutiveHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=7,
    )


    body_style = ParagraphStyle(
        "ExecutiveBody",
        parent=styles["BodyText"],
        fontSize=8.8,
        leading=12.5,
        spaceAfter=5,
    )


    bullet_style = ParagraphStyle(
        "ExecutiveBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=5,
    )


    small_style = ParagraphStyle(
        "ExecutiveSmall",
        parent=body_style,
        fontSize=7.5,
        leading=10,
    )


    story = []


    # ========================================================
    # PDF PAGE 1 — EXECUTIVE SUMMARY
    # ========================================================

    generated_at = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )


    story.append(
        Paragraph(
            "Olist E-Commerce Ecosystem Analytics",
            title_style,
        )
    )


    story.append(
        Paragraph(
            "Executive Performance Report",
            subtitle_style,
        )
    )


    story.append(
        Paragraph(
            f"<b>Generated:</b> {generated_at}",
            body_style,
        )
    )


    story.append(
        Paragraph(
            "<b>Analysis Period:</b> "
            f"{start_date.strftime('%d %b %Y')} — "
            f"{end_date.strftime('%d %b %Y')}",
            body_style,
        )
    )


    # ========================================================
    # FILTER SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Applied Filter Summary",
            heading_style,
        )
    )


    payment_text = (
        ", ".join(
            map(str, payments)
        )
        if payments
        else "None"
    )


    filter_table_data = [
        [
            "Filter",
            "Selected Value",
        ],
        [
            "Date Range",
            f"{start_date.strftime('%d %b %Y')} — "
            f"{end_date.strftime('%d %b %Y')}",
        ],
        [
            "Customer State",
            str(state),
        ],
        [
            "Payment Type",
            payment_text,
        ],
        [
            "Filtered Records",
            f"{len(data):,}",
        ],
    ]


    filter_table = Table(
        filter_table_data,
        colWidths=[
            48 * mm,
            122 * mm,
        ],
    )


    filter_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#E7E7E7"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        filter_table
    )


    # ========================================================
    # KPI TABLE
    # ========================================================

    story.append(
        Paragraph(
            "Executive KPI Scorecard",
            heading_style,
        )
    )


    kpi_table_data = [
        [
            "Metric",
            "Current Value",
        ],
        [
            "Total GMV",
            currency(
                data["total_price"].sum()
            ),
        ],
        [
            "Total Orders",
            f"{data['order_id'].nunique():,}",
        ],
        [
            "Customers",
            f"{data['customer_unique_id'].nunique():,}",
        ],
        [
            "Active Sellers",
            f"{data['seller_id'].nunique():,}",
        ],
        [
            "Average Order Value",
            currency(
                (
                    data["total_price"].sum()
                    /
                    data["order_id"].nunique()
                )
                if data["order_id"].nunique()
                else np.nan
            ),
        ],
        [
            "Average Review",
            score(
                data["review_score"].mean()
            ),
        ],
        [
            "Average Delivery",
            (
                f"{data['actual_delivery_time_days'].mean():.1f} days"
                if not data[
                    "actual_delivery_time_days"
                ].dropna().empty
                else "N/A"
            ),
        ],
        [
            "Late Delivery Rate",
            percent(
                data["is_late"].mean()
                * 100
            ),
        ],
        [
            "Repeat Buyer Rate",
            percent(
                (
                    (
                        data
                        .groupby(
                            "customer_unique_id"
                        )["order_id"]
                        .nunique()
                        > 1
                    ).mean()
                    * 100
                )
            ),
        ],
    ]


    kpi_table = Table(
        kpi_table_data,
        colWidths=[
            75 * mm,
            95 * mm,
        ],
    )


    kpi_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#E7E7E7"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        kpi_table
    )


    # ========================================================
    # MANAGEMENT INTERPRETATION
    # ========================================================

    story.append(
        Paragraph(
            "Management Interpretation",
            heading_style,
        )
    )


    pdf_gmv = data[
        "total_price"
    ].sum()

    pdf_orders = data[
        "order_id"
    ].nunique()

    pdf_customers = data[
        "customer_unique_id"
    ].nunique()

    pdf_review = data[
        "review_score"
    ].mean()

    pdf_delivery = data[
        "actual_delivery_time_days"
    ].mean()

    pdf_late = (
        data["is_late"].mean()
        * 100
    )


    management_text = (
        f"The selected dataset segment generated "
        f"<b>{currency(pdf_gmv)}</b> in GMV across "
        f"<b>{pdf_orders:,}</b> orders and "
        f"<b>{pdf_customers:,}</b> unique customers. "
        f"Average customer satisfaction was "
        f"<b>{pdf_review:.2f}★</b>, while the average "
        f"observed delivery duration was "
        f"<b>{pdf_delivery:.1f} days</b>. "
        f"The filtered late-delivery rate was "
        f"<b>{pdf_late:.1f}%</b>."
    )


    story.append(
        Paragraph(
            management_text,
            body_style,
        )
    )


    # ========================================================
    # PAGE BREAK
    # ========================================================

    story.append(
        PageBreak()
    )


    # ========================================================
    # PDF PAGE 2 — STRATEGIC ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "Strategic Analysis & Recommendations",
            title_style,
        )
    )


    story.append(
        Paragraph(
            "Data-driven management priorities derived from the active dashboard filters.",
            subtitle_style,
        )
    )


    # ========================================================
    # TOP CATEGORY
    # ========================================================

    pdf_category = (
        data
        .groupby(
            "product_category_name_english"
        )["total_price"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    if not pdf_category.empty:

        pdf_top_category = (
            pdf_category.index[0]
        )

        pdf_top_category_value = (
            pdf_category.iloc[0]
        )

    else:

        pdf_top_category = "N/A"
        pdf_top_category_value = 0


    # ========================================================
    # TOP STATE
    # ========================================================

    pdf_state = (
        data
        .groupby(
            "customer_state"
        )["order_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    if not pdf_state.empty:

        pdf_top_state = pdf_state.index[0]
        pdf_top_state_orders = pdf_state.iloc[0]

    else:

        pdf_top_state = "N/A"
        pdf_top_state_orders = 0


    # ========================================================
    # SELLER CONCENTRATION
    # ========================================================

    pdf_sellers = (
        data
        .groupby("seller_id")[
            "total_price"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    if not pdf_sellers.empty:

        seller_total = pdf_sellers.sum()

        pdf_top1 = (
            pdf_sellers.head(1).sum()
            / seller_total
            * 100
        )

        pdf_top5 = (
            pdf_sellers.head(5).sum()
            / seller_total
            * 100
        )

    else:

        pdf_top1 = 0
        pdf_top5 = 0


    # ========================================================
    # STRATEGIC SNAPSHOT TABLE
    # ========================================================

    story.append(
        Paragraph(
            "Strategic Snapshot",
            heading_style,
        )
    )


    strategic_data = [
        [
            "Dimension",
            "Finding",
        ],
        [
            "Leading Category",
            f"{title_case(pdf_top_category)} "
            f"({currency(pdf_top_category_value)})",
        ],
        [
            "Leading State",
            f"{pdf_top_state} "
            f"({pdf_top_state_orders:,} orders)",
        ],
        [
            "Top Seller Share",
            percent(pdf_top1),
        ],
        [
            "Top 5 Seller Share",
            percent(pdf_top5),
        ],
        [
            "Freight / GMV",
            percent(
                (
                    data["total_freight"].sum()
                    /
                    data["total_price"].sum()
                    * 100
                )
                if data["total_price"].sum() > 0
                else 0
            ),
        ],
    ]


    strategic_table = Table(
        strategic_data,
        colWidths=[
            65 * mm,
            105 * mm,
        ],
    )


    strategic_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#E7E7E7"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )


    story.append(
        strategic_table
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "Priority Business Recommendations",
            heading_style,
        )
    )


    pdf_recommendations = []


    if pdf_late >= 15:

        pdf_recommendations.append(
            "1. <b>Logistics:</b> Reduce late deliveries by identifying "
            "high-volume delayed sellers, routes and fulfillment bottlenecks."
        )

    else:

        pdf_recommendations.append(
            "1. <b>Logistics:</b> Maintain current delivery discipline "
            "while monitoring delayed orders at seller and geographic levels."
        )


    if not pd.isna(pdf_review):

        if pdf_review < 4:

            pdf_recommendations.append(
                "2. <b>Customer Experience:</b> The average rating is "
                "below the 4.0★ benchmark. Investigate the operational "
                "drivers of low ratings, especially delivery and freight friction."
            )

        else:

            pdf_recommendations.append(
                "2. <b>Customer Experience:</b> Satisfaction is at or above "
                "the 4.0★ benchmark. Preserve service quality while "
                "improving low-performing categories."
            )


    pdf_customer_counts = (
        data
        .groupby(
            "customer_unique_id"
        )["order_id"]
        .nunique()
    )


    pdf_repeat_rate = (
        (pdf_customer_counts > 1).mean()
        * 100
        if len(pdf_customer_counts)
        else 0
    )


    if pdf_repeat_rate < 25:

        pdf_recommendations.append(
            "3. <b>Retention:</b> Repeat purchasing is relatively limited. "
            "Build customer lifecycle programs targeting one-time buyers "
            "with relevant products and post-purchase engagement."
        )

    else:

        pdf_recommendations.append(
            "3. <b>Retention:</b> Repeat purchasing represents a meaningful "
            "customer base. Protect retention with loyalty and personalization."
        )


    if pdf_top5 >= 50:

        pdf_recommendations.append(
            "4. <b>Seller Risk:</b> Revenue concentration among the top "
            "five sellers is high. Establish dependency monitoring and "
            "seller diversification initiatives."
        )

    else:

        pdf_recommendations.append(
            "4. <b>Seller Risk:</b> Seller concentration is relatively "
            "controlled, but high-value sellers should remain under review."
        )


    pdf_freight_ratio = (
        data["total_freight"].sum()
        /
        data["total_price"].sum()
        * 100
        if data["total_price"].sum() > 0
        else 0
    )


    if pdf_freight_ratio >= 20:

        pdf_recommendations.append(
            "5. <b>Freight Economics:</b> Freight represents a significant "
            "portion of GMV. Review shipping pricing, destination economics "
            "and opportunities for logistics optimization."
        )

    else:

        pdf_recommendations.append(
            "5. <b>Freight Economics:</b> Freight exposure appears "
            "relatively controlled. Continue monitoring high-friction "
            "customer segments."
        )


    for recommendation in pdf_recommendations:

        story.append(
            Paragraph(
                f"• {recommendation}",
                bullet_style,
            )
        )


    # ========================================================
    # OPPORTUNITY / RISK SECTION
    # ========================================================

    story.append(
        Paragraph(
            "Key Risks & Opportunities",
            heading_style,
        )
    )


    risks = []


    if pdf_late >= 15:

        risks.append(
            "High delivery lateness can negatively affect customer satisfaction "
            "and future purchasing behavior."
        )


    if pdf_repeat_rate < 25:

        risks.append(
            "Low repeat-buyer penetration represents a significant "
            "customer lifetime value opportunity."
        )


    if pdf_top5 >= 50:

        risks.append(
            "High seller concentration creates marketplace dependency risk."
        )


    if pdf_freight_ratio >= 20:

        risks.append(
            "High freight burden may create price sensitivity and "
            "customer experience friction."
        )


    if not risks:

        risks.append(
            "No critical threshold-based risk was detected from the "
            "selected KPI indicators. Continue proactive monitoring."
        )


    for risk in risks:

        story.append(
            Paragraph(
                f"• {risk}",
                bullet_style,
            )
        )


    # ========================================================
    # FINAL EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Executive Conclusion",
            heading_style,
        )
    )


    conclusion = (
        f"The selected Olist segment currently represents "
        f"{currency(pdf_gmv)} of GMV across "
        f"{pdf_orders:,} orders. "
        f"Management attention should be prioritized around "
        f"delivery reliability, customer retention, seller concentration "
        f"and freight economics. "
        f"The leading product category is "
        f"<b>{title_case(pdf_top_category)}</b>, while "
        f"<b>{pdf_top_state}</b> represents the largest customer-state "
        f"order concentration."
    )


    story.append(
        Paragraph(
            conclusion,
            body_style,
        )
    )


    # ========================================================
    # REPORT FOOTNOTE
    # ========================================================

    story.append(
        Spacer(
            1,
            8,
        )
    )


    story.append(
        Paragraph(
            "This report is dynamically generated from the current "
            "dashboard filter state. All metrics represent the filtered "
            "dataset available at report generation time.",
            small_style,
        )
    )


    # ========================================================
    # BUILD
    # ========================================================

    document.build(
        story
    )


    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GENERATE CURRENT PDF
# ============================================================

pdf_bytes = build_pdf_report(
    data=filtered_df,
    start_date=start_date,
    end_date=end_date,
    state=selected_state,
    payments=selected_payment_types,
)


# ============================================================
# PDF DOWNLOAD
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    "📄 10 — Executive PDF Report"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Generate and download a management-ready report using the exact filters currently applied."
    "</div>",
    unsafe_allow_html=True,
)


pdf_c1, pdf_c2, pdf_c3 = st.columns(
    [1, 2, 1]
)


with pdf_c2:

    st.download_button(
        label=(
            "📥 Download Olist Executive Report"
        ),
        data=pdf_bytes,
        file_name=REPORT_FILE_NAME,
        mime="application/pdf",
        use_container_width=True,
        key="download_executive_pdf",
    )


st.caption(
    "The downloaded PDF contains the current filter summary, KPI scorecard, "
    "strategic findings, risks and business recommendations."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Olist E-Commerce Ecosystem Analytics • "
    "Streamlit + Pandas + Plotly + ReportLab"
)