import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Data Gathering
customers_df = pd.read_csv(os.path.join(current_dir, "../data/customers_dataset.csv"))
geolocation_df = pd.read_csv(os.path.join(current_dir, "../data/geolocation_dataset.csv"))
orders_df = pd.read_csv(os.path.join(current_dir, "../data/orders_dataset.csv"))
order_items_df = pd.read_csv(os.path.join(current_dir, "../data/order_items_dataset.csv"))
order_payments_df = pd.read_csv(os.path.join(current_dir, "../data/order_payments_dataset.csv"))
order_reviews_df = pd.read_csv(os.path.join(current_dir, "../data/order_reviews_dataset.csv"))
products_df = pd.read_csv(os.path.join(current_dir, "../data/products_dataset.csv"))
product_category_name_translations_df = pd.read_csv(os.path.join(current_dir, "../data/product_category_name_translation.csv"))
sellers_df = pd.read_csv(os.path.join(current_dir, "../data/sellers_dataset.csv"))

# Data Cleaning
datetime_columns = ['review_creation_date', 'review_answer_timestamp']
for column in datetime_columns:
    order_reviews_df[column] = pd.to_datetime(order_reviews_df[column])

products_df = pd.merge(
    products_df[['product_id', 'product_category_name']],
    product_category_name_translations_df,
    how='left',
    on='product_category_name'
)
products_df["product_category_name"].fillna("other", inplace=True)

geolocation_df.drop_duplicates(inplace=True)
orders_df.dropna(subset=["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"], inplace=True)
datetime_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for column in datetime_columns:
    orders_df[column] = pd.to_datetime(orders_df[column])

order_reviews_df.drop(columns=["review_comment_title", "review_comment_message"], inplace=True)

# Exploratory Data Analysis
orders_df['year_of_purchase'] = orders_df['order_purchase_timestamp'].dt.year
order_per_year_df = orders_df.groupby(by='year_of_purchase').agg({
    'order_id': 'count'
}).reset_index().rename(columns={'order_id': 'total_order'})

order_customer_df = pd.merge(orders_df, customers_df, on='customer_id', how='left')
order_per_state_df = order_customer_df.groupby('customer_state').agg({
    'order_id': 'count'
}).reset_index().rename(columns={'customer_state': 'state', 'order_id': 'total_order'})

customer_status_df = pd.DataFrame()
customer_status_df["customer_id"] = customers_df["customer_id"]
customer_status_df["status"] = customers_df["customer_id"].isin(orders_df["customer_id"]).map({True: "Active", False: "Inactive"})
customer_status_df = customer_status_df.groupby('status').agg({
    'customer_id': 'count'
}).reset_index().rename(columns={'customer_id': 'total_customer'})

review_score_df = order_reviews_df.groupby('review_score').agg({
    'review_id': 'count'
}).reset_index().rename(columns={'review_id': 'total_review'})

order_items_product_df = pd.merge(
    order_items_df,
    products_df,
    how="left",
    on="product_id"
)
order_product_category_df = order_items_product_df.groupby('product_category_name_english').agg({
    'order_id': 'count'
}).reset_index().rename(columns={'order_id': 'total_order'})

# Sidebar Filters
st.sidebar.header("Filter Options")
date_range = st.sidebar.date_input("Select Date Range", 
                                   value=(orders_df['order_purchase_timestamp'].min(), orders_df['order_purchase_timestamp'].max()))

# Apply Date Filter
filtered_orders = orders_df[
    (orders_df['order_purchase_timestamp'] >= pd.to_datetime(date_range[0])) &
    (orders_df['order_purchase_timestamp'] <= pd.to_datetime(date_range[1]))
]

# Streamlit Dashboard
st.title("E-Commerce Dashboard")

st.header("Key Performance Indicators (KPI)")

# KPI metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Customers", value=customers_df.customer_id.nunique())
with col2:
    st.metric(label="Total Orders", value=orders_df.shape[0])
with col3:
    st.metric(label="Average Order Value", value=f"${order_items_df.groupby('order_id')['price'].sum().mean():,.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric(label="Total Sellers", value=sellers_df.seller_id.nunique())
with col5:
    st.metric(label="Total Reviews", value=order_reviews_df.shape[0])
with col6:
    st.metric(label="Average Review Score", value=f"{order_reviews_df['review_score'].mean():.2f}")

# Visualisasi
st.header("Pertanyaan Bisnis")

# P1: Jumlah Order Berdasarkan State
st.subheader("P1: Jumlah Order Berdasarkan State")
order_state_count = filtered_orders.groupby('customer_state')['order_id'].count().reset_index()
order_state_count.columns = ['State', 'Total Orders']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=order_state_count.sort_values('Total Orders', ascending=False), 
            x='State', y='Total Orders', palette=['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(order_state_count))], ax=ax)
plt.title("Jumlah Order Berdasarkan State")
plt.xticks(rotation=45)
st.pyplot(fig)

# P2: Pertumbuhan Order per Tahun
st.subheader("P2: Pertumbuhan Order per Tahun")
filtered_orders['year'] = filtered_orders['order_purchase_timestamp'].dt.year
orders_per_year = filtered_orders.groupby('year')['order_id'].count().reset_index()
orders_per_year.columns = ['Year', 'Total Orders']

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=orders_per_year, x='Year', y='Total Orders', marker="o", ax=ax)
plt.title("Pertumbuhan Order per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Order")
st.pyplot(fig)

# P3: Customer Active vs Inactive
st.subheader("P3: Customer Active vs Inactive")
customer_status = pd.DataFrame({
    "Status": ["Active", "Inactive"],
    "Count": [filtered_orders['customer_id'].nunique(), 
              customers_df['customer_id'].nunique() - filtered_orders['customer_id'].nunique()]
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.pie(customer_status['Count'], labels=customer_status['Status'], autopct='%1.1f%%', colors=["#5B9BD5", "#A2C4E4"])
plt.title("Customer Active vs Inactive")
st.pyplot(fig)
