import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION & DATA LOADING ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, cleans, and prepares the retail data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    # 1. Data Type Conversion & Cleaning
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Date']) 
    
    # Merge holiday categories (Public 'a' and Easter 'b')
    df['StateHoliday'] = df['StateHoliday'].astype(str).str.strip().replace({
        '0': 'No Holiday',
        'a': 'Major Public Holiday',   
        'b': 'Major Public Holiday',   
        'c': 'Christmas' 
    })

    # Ensure Order_Demand is numeric and handle potential zeros for log scale
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce').fillna(0)
    
    # Filter out closed stores for core demand analysis (Open = 1)
    df_active = df[df['Open'] == 1].copy()
    
    # Feature Engineering for Time Series
    df_active['YearMonth'] = df_active['Date'].dt.to_period('M').astype(str)
    
    # Add Log Demand for better visualization of highly skewed data
    # We use log(1+x) to handle zero values gracefully (Log Demand)
    df_active['Log_Demand'] = np.log1p(df_active['Order_Demand'])
    
    return df, df_active

# Set up the Streamlit page
st.set_page_config(layout="wide", page_title="Retail Sales Demand Analysis")

# Load data
df_raw, df = load_and_preprocess_data('retail_data.csv')

if df.empty:
    st.stop()
    
st.title("🛒 Retail Stock Demand Report")
st.markdown("This report analyzes how many products customers order each day, helping us plan stock levels and manage costs.")
st.markdown("---")

# --- INTERACTIVE FILTERING (Sidebar) ---

with st.sidebar:
    st.header("Report Filters")
    
    # Warehouse Selector
    selected_warehouse = st.selectbox(
        "Filter by Warehouse:",
        options=['All'] + list(df['Warehouse'].unique())
    )
    
    # Apply warehouse filter
    if selected_warehouse != 'All':
        df_filtered = df[df['Warehouse'] == selected_warehouse]
    else:
        df_filtered = df.copy()


# --- I. KEY PERFORMANCE INDICATORS (KPI Snapshot) ---

st.header("I. KPI Snapshot: Overall Business Health")
col1, col2, col3, col4 = st.columns(4)

total_demand = df_filtered['Order_Demand'].sum()
unique_products = df_filtered['Product_Code'].nunique()
unique_categories = df_filtered['Product_Category'].nunique()
avg_demand_per_order = df_filtered['Order_Demand'].mean().round(2)

col1.metric("Total Products Needed (Volume)", f"{total_demand:,.0f}")
col2.metric("Unique Products", unique_products)
col3.metric("Unique Categories", unique_categories)
col4.metric("Average Order Size", f"{avg_demand_per_order:,.0f}")

st.markdown("""
**Description:** These numbers show the **total scale** of the business. The **Total Products Needed** tells us the stock required and the **Average Order Size** helps set standard reorder quantities.
""")
st.markdown("---")


# --- II. DEMAND TREND & VOLATILITY ---

st.header("II. Demand Trend & Stability")

# 1. Time Series Plot (Daily/Monthly)
demand_trend = df_filtered.groupby('Date')['Order_Demand'].sum().reset_index()

fig_trend = px.line(
    demand_trend,
    x='Date',
    y='Order_Demand',
    title='Daily Total Products Ordered Over Time'
)
fig_trend.update_layout(yaxis_title="Total Products Ordered", xaxis_title="Date")
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""
Check the plot for **seasonal ups and downs**. Find **spikes** which often mean a big sales happened in plot. A very bumpy line means stock forecasting will be difficult. We can see that someday have no product ordered. So next we will see what we can do for manage stock in each Warehouses.
""")

# 2. Demand Distribution (Box Plot for Volatility)
st.subheader("Order Size Stability Check")

demand_per_unique_order = df_filtered.groupby(['Date', 'Warehouse', 'Product_Code'])['Order_Demand'].mean().reset_index()
fig_box = px.box(
    demand_per_unique_order, 
    y="Order_Demand", 
    x="Warehouse",
    log_y=True, 
    title="Order Size Stability by Warehouse"
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("""
This chart show order stability. **Outlier points (dots above the lines)** mean you have many huge, unpredictable orders. If the vertical boxes or lines are very long, the warehouse has **unstable order sizes** and needs very safety stock.
""")
st.markdown("---")


# --- III. CATEGORICAL BREAKDOWN (Where is the demand?) ---

st.header("III. Key Demand Areas")

col_cat_1, col_cat_2 = st.columns(2)

with col_cat_1:
    st.subheader("Warehouse Performance")
    # 3. Warehouse Performance Bar Chart (Total vs. Average)
    warehouse_demand = df_filtered.groupby('Warehouse')['Order_Demand'].agg(['sum', 'mean']).reset_index()
    
    fig_wh_total = px.bar(
        warehouse_demand,
        x='Warehouse',
        y='sum',
        title="Total Products Ordered by Warehouse",
        text_auto='.2s'
    )
    st.plotly_chart(fig_wh_total, use_container_width=True)
    st.markdown("""
    **Insight:** The tallest bar shows **the largest volume warehouse**. This warehouse needs the most attention for logistics and inventory investment.
    """)

with col_cat_2:
    st.subheader("Top Product Categories")
    # 4. Product Category Bar Chart (Top N)
    category_demand = df_filtered.groupby('Product_Category')['Order_Demand'].sum().nlargest(10).reset_index()
    
    fig_cat = px.bar(
        category_demand,
        x='Product_Category',
        y='Order_Demand',
        title="Top 10 Product Categories have the most Ordered",
        text_auto='.2s'
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown("""
    **Insight:** These **Top 10 categories** drive the main of orders demand. Focus on getting the stock right for these items especially **Category_019** that have over 600 million orders demand, as mistakes here cost the most. In conclude we should focus a lot in Warehouse_J and Category_019
    """)

st.markdown("---")


# --- IV. DEMAND DRIVER ANALYSIS (What influences demand?) ---

st.header("IV. What Causes Demand to Change?")

col_driver_1, col_driver_2 = st.columns(2)

with col_driver_1:
    st.subheader("Promotion Impact")
    # 5. Promotion Effectiveness
    promo_demand = df_filtered.groupby('Promo')['Order_Demand'].mean().reset_index()
    promo_demand['Promo'] = promo_demand['Promo'].replace({0: 'No Promo', 1: 'Promo Running'})
    
    fig_promo = px.bar(
        promo_demand,
        x='Promo',
        y='Order_Demand',
        title='Average Order Size: With vs. Without Promo',
        text_auto='.2s'
    )
    st.plotly_chart(fig_promo, use_container_width=True)
    st.markdown("""
    **Insight:** If the **'Promo Running'** bar is much taller, promotions successfully increase the average order demand. In this case the **'Promo running'** bar is not taller than **'No Promo'** bar so we should adjust our Promotions more attractive customers.
    """)

with col_driver_2:
    st.subheader("Holiday Impact")
    # 6. Holiday Impact (Merged)
    holiday_demand = df_filtered.groupby('StateHoliday')['Order_Demand'].mean().reset_index().sort_values(by='Order_Demand', ascending=False)
    
    fig_holiday = px.bar(
        holiday_demand,
        x='StateHoliday',
        y='Order_Demand',
        title="Average Order Size by Holiday Type",
        color='StateHoliday',
        text_auto='.2s'
    )
    st.plotly_chart(fig_holiday, use_container_width=True)
    st.markdown("""
    **Insight:** Compare holidays with **'No Holiday'**. Major Public Holidays usually cause a **spike in demand** as customers shop more. But from this chart we should set higher stock levels before the holiday period begins. --That show customers, that lives in warehouses surrounding area **except** Warehouse_C (see in next chart), have usually brought products before their holiday.
    """)

# 7. State Holiday Impact (Grouped Bar Chart)
st.subheader("Holiday Impact by Warehouse")

# Group by Warehouse and the detailed StateHoliday type
state_holiday_impact = df_filtered.groupby(['Warehouse', 'StateHoliday'])['Order_Demand'].mean().reset_index()

# Create the grouped bar chart
fig_state_holiday = px.bar(
    state_holiday_impact,
    x='Warehouse',
    y='Order_Demand',
    color='StateHoliday',
    barmode='group',
    title='Average Order Size on Holidays, Grouped by Warehouse',
    # Manually set the order of holiday types for better visualization
    category_orders={"StateHoliday": ["No Holiday", "Major Public Holiday", "Christmas"]}
)

st.plotly_chart(fig_state_holiday, use_container_width=True)
st.markdown("""
**Insight:** As you can see above chart shows which warehouses have the biggest demand during holidays. Some warehouses may be more affected than others and require special holiday stock plans.
""")
st.markdown("""
**Analysis:** In holidays we have to increase stock more than normal days in every warehouses, But only **Warehouse_C** we should decrease stock more than normal days.
""")
st.markdown("---")


# --- V. EXTERNAL FACTOR CORRELATION (UPDATED RAW DATA SCATTER - NO LOG SCALE) ---

st.header("V. Fuel Price vs. Demand (Raw Data)")
st.markdown("This plot shows the relationship between the **individual order demand** and the **fuel price** on that day using the **raw demand values**. This allows us to see the true size of the largest orders.")

# 8. Petrol Price Correlation (Scatter Plot) using RAW data
# We use df_filtered directly (no aggregation)
fig_scatter = px.scatter(
    df_filtered,
    x='Petrol_price',       # Raw individual petrol price
    y='Order_Demand',       # RAW (normal) order demand
    trendline="ols",        # Add a linear trendline 
    opacity=0.3,            # Set opacity to 0.3 to better visualize dense areas
    title='Individual Order Demand (Raw Scale) vs. Raw Fuel Price',
    labels={
        'Petrol_price': 'Raw Petrol Price', 
        'Order_Demand': 'Order Demand (Raw Scale)' # Label changed from Log Scale
    },
    hover_data=['Order_Demand', 'Date', 'Warehouse']
)

st.plotly_chart(fig_scatter, use_container_width=True)

# Calculate the correlation based on raw data for the metric
correlation = df_filtered['Petrol_price'].corr(df_filtered['Order_Demand']).round(3)
st.info(f"The Pearson correlation coefficient between **Raw Fuel Price** and **Raw Order Demand** is **{correlation}**.")
st.markdown("""
**Description:** The majority of the data points will be clustered near the bottom (0-10,000 range), making the high-demand outliers clearly visible on the large scale.
""")
st.markdown("""
**Analysis:** From above dot plot, A correlation score close to **zero** means Orders demand is not significantly releated to Fuel price.
""")