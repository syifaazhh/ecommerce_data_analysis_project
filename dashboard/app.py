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

st.subheader("P1: Kategori Produk Terpopuler Berdasarkan Jumlah Pesanan")
top_categories = order_product_category_df.sort_values(by='total_order', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=top_categories,
    x='total_order',
    y='product_category_name_english',
    palette=['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(top_categories))],
    ax=ax
)
plt.title("Top Kategori Produk Berdasarkan Jumlah Pesanan")
ax.set_xlabel('')
ax.set_ylabel('')
st.pyplot(fig)

st.subheader("P2: Bagaimana pola pembelian harian, mingguan, atau bulanan? Apakah ada waktu tertentu dengan volume penjualan yang tinggi?")
st.write("a. Tren Penjualan Tahunan")
# Hitung jumlah pesanan per tahun
orders_df['purchase_year'] = orders_df['order_purchase_timestamp'].dt.year
orders_per_year = orders_df.groupby('purchase_year')['order_id'].count().reset_index()
orders_per_year.columns = ['purchase_year', 'total_orders']

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=orders_per_year, x='purchase_year', y='total_orders', marker='o', ax=ax)
ax.set_title('Total Orders per Year')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([2016, 2017, 2018])
plt.tight_layout()
st.pyplot(fig)

st.write("b. Tren Penjualan Bulanan")
# Hitung jumlah pesanan per bulan
orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
orders_per_month = orders_df.groupby('purchase_month')['order_id'].count().reset_index()
orders_per_month.columns = ['purchase_month', 'total_orders']

# Mapping angka bulan menjadi nama bulan
month_names = {
    1: 'January', 2: 'Februari=y', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
orders_per_month['purchase_month'] = orders_per_month['purchase_month'].map(month_names)

# Plot
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(data=orders_per_month, x='purchase_month', y='total_orders', marker='o', ax=ax)
ax.set_title('Total Orders per Month')
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.subheader("P3: Top 10 Kategori Produk Berdasarkan Total Pendapatan")
st.write("Visualisasi kategori produk dengan total pendapatan tertinggi.")

# Menggabungkan data untuk perhitungan total pendapatan
merged_df = pd.merge(order_items_df, products_df, on='product_id', how='left')
merged_df = pd.merge(merged_df, product_category_name_translations_df, on='product_category_name', how='left')

# Mengganti nama kolom hasil merge untuk konsistensi
merged_df.rename(columns={'product_category_name_translations': 'product_category_name_english'}, inplace=True)

# Pastikan tidak ada nilai NaN di kolom 'price' atau 'order_item_id'
merged_df['price'] = merged_df['price'].fillna(0)
merged_df['order_item_id'] = merged_df['order_item_id'].fillna(0)

# Menghitung total pendapatan
merged_df['total_revenue'] = merged_df['price'] * merged_df['order_item_id']

# Menghitung total pendapatan per kategori produk
revenue_per_category = merged_df.groupby('product_category_name_english')['total_revenue'].sum().reset_index()
revenue_per_category = revenue_per_category.sort_values(by='total_revenue', ascending=False)

# Menentukan warna khusus untuk kategori dengan pendapatan tertinggi
revenue_per_category['color'] = ['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(revenue_per_category))]

# Membatasi hanya 10 kategori teratas
top_10_revenue = revenue_per_category.head(10)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='total_revenue',
    y='product_category_name_english',
    data=top_10_revenue,
    palette=top_10_revenue['color'],
    ax=ax
)
ax.set_title('Top 10 Product Categories by Total Revenue')
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
