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

st.subheader("P3: Produk dengan Total Pendapatan Tertinggi")

# Menghitung total pendapatan per produk
revenue_per_product = order_items_df.groupby('product_id')['price'].sum().reset_index()
revenue_per_product = pd.merge(revenue_per_product, products_df, on='product_id', how='left')
revenue_per_product = pd.merge(revenue_per_product, product_category_name_translations_df, on='product_category_name', how='left')
revenue_per_product.rename(columns={'price': 'total_revenue'}, inplace=True)
top_10_products = revenue_per_product.sort_values(by='total_revenue', ascending=False).head(10)

# Menentukan warna
top_10_products['color'] = ['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(top_10_products))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='total_revenue',
    y='product_category_name_english',
    data=top_10_products,
    palette=top_10_products['color'],
    ax=ax
)
ax.set_title('Top 10 Produk Berdasarkan Total Pendapatan')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P4: Dari Daerah Mana Pelanggan Terbanyak Berasal?")

# Menghitung jumlah pelanggan unik per state
customer_by_state = customers_df.groupby('customer_state')['customer_id'].nunique().reset_index()
customer_by_state.rename(columns={'customer_id': 'total_customers'}, inplace=True)
top_10_states = customer_by_state.sort_values(by='total_customers', ascending=False).head(10)

# Menentukan warna
top_10_states['color'] = ['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(top_10_states))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='total_customers',
    y='customer_state',
    data=top_10_states,
    palette=top_10_states['color'],
    ax=ax
)
ax.set_title('Top 10 State Berdasarkan Jumlah Pelanggan')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P5: Rasio Pelanggan Baru vs Pelanggan Lama")

# Menentukan status pelanggan
customers_df['is_returning'] = customers_df['customer_id'].isin(orders_df['customer_id']).map({True: 'Returning', False: 'New'})
customer_status = customers_df.groupby('is_returning')['customer_id'].count().reset_index()
customer_status.rename(columns={'customer_id': 'total_customers'}, inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#5B9BD5' if label == 'Returning' else '#A2C4E4' for label in customer_status['is_returning']]
ax.pie(customer_status['total_customers'], labels=customer_status['is_returning'], autopct='%1.1f%%', colors=colors)
ax.set_title('Rasio Pelanggan Baru vs Pelanggan Lama')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P6: Metode Pembayaran Paling Sering Digunakan")

# Menghitung jumlah penggunaan metode pembayaran
payment_methods = order_payments_df['payment_type'].value_counts().reset_index()
payment_methods.columns = ['payment_type', 'count']

# Menentukan warna
payment_methods['color'] = ['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(payment_methods))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='count',
    y='payment_type',
    data=payment_methods,
    palette=payment_methods['color'],
    ax=ax
)
ax.set_title('Metode Pembayaran yang Paling Sering Digunakan')
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P7: Hubungan Metode Pembayaran dan Waktu Pengiriman")

# Menghitung waktu pengiriman
orders_df['delivery_time'] = (orders_df['order_delivered_customer_date'] - orders_df['order_purchase_timestamp']).dt.days
orders_payments = pd.merge(order_payments_df, orders_df[['order_id', 'delivery_time']], on='order_id', how='left')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    x='payment_type',
    y='delivery_time',
    data=orders_payments,
    palette=['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(orders_payments['payment_type'].unique()))],
    ax=ax
)
ax.set_title('Hubungan Metode Pembayaran dan Waktu Pengiriman')
ax.set_xlabel('')
ax.set_ylabel('Waktu Pengiriman (Hari)')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P8: Distribusi Rating Ulasan Pelanggan")

# Menghitung jumlah ulasan berdasarkan rating
review_distribution = order_reviews_df['review_score'].value_counts().reset_index()
review_distribution.columns = ['review_score', 'count']
review_distribution = review_distribution.sort_values(by='review_score', ascending=True)

# Menentukan warna
review_distribution['color'] = ['#5B9BD5' if i == review_distribution['count'].idxmax() else '#A2C4E4' for i in range(len(review_distribution))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='review_score',
    y='count',
    data=review_distribution,
    palette=review_distribution['color'],
    ax=ax
)
ax.set_title('Distribusi Rating Ulasan Pelanggan')
ax.set_xlabel('Rating')
ax.set_ylabel('Jumlah Ulasan')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P9: Perbedaan Waktu Pengiriman Berdasarkan Ulasan Positif dan Negatif")

# Menentukan sentimen berdasarkan skor ulasan
orders_reviews = pd.merge(order_reviews_df, orders_df, on='order_id', how='left')
orders_reviews['delivery_time'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_purchase_timestamp']).dt.days
orders_reviews['sentiment'] = orders_reviews['review_score'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    x='sentiment',
    y='delivery_time',
    data=orders_reviews,
    palette=['#5B9BD5', '#A2C4E4'],
    ax=ax
)
ax.set_title('Waktu Pengiriman Berdasarkan Sentimen Ulasan')
ax.set_xlabel('Sentimen Ulasan')
ax.set_ylabel('Waktu Pengiriman (Hari)')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P10: Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk")

# Menghitung rata-rata waktu pengiriman berdasarkan kategori produk
order_items_product_df['delivery_time'] = (
    pd.to_datetime(order_items_product_df['order_delivered_customer_date']) -
    pd.to_datetime(order_items_product_df['order_purchase_timestamp'])
).dt.days
avg_delivery = order_items_product_df.groupby('product_category_name_english')['delivery_time'].mean().reset_index()
avg_delivery = avg_delivery.sort_values(by='delivery_time', ascending=False).head(10)

# Menentukan warna
avg_delivery['color'] = ['#5B9BD5' if i == avg_delivery['delivery_time'].idxmax() else '#A2C4E4' for i in range(len(avg_delivery))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='delivery_time',
    y='product_category_name_english',
    data=avg_delivery,
    palette=avg_delivery['color'],
    ax=ax
)
ax.set_title('Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk')
ax.set_xlabel('Rata-Rata Waktu Pengiriman (Hari)')
ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.subheader("P11: Keterlambatan Pengiriman Berdasarkan Kategori Produk")

# Menentukan keterlambatan pengiriman
order_items_product_df['is_late'] = order_items_product_df['delivery_time'] > (
    pd.to_datetime(order_items_product_df['order_estimated_delivery_date']) -
    pd.to_datetime(order_items_product_df['order_purchase_timestamp'])
).dt.days
late_delivery = order_items_product_df[order_items_product_df['is_late']].groupby('product_category_name_english').size().reset_index()
late_delivery.columns = ['product_category_name_english', 'late_count']
late_delivery = late_delivery.sort_values(by='late_count', ascending=False).head(10)

# Menentukan warna
late_delivery['color'] = ['#5B9BD5' if i == 0 else '#A2C4E4' for i in range(len(late_delivery))]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='late_count',
    y='product_category_name_english',
    data=late_delivery,
    palette=late_delivery['color'],
    ax=ax
)
ax.set_title('Keterlambatan Pengiriman Berdasarkan Kategori Produk')
ax.set_xlabel('Jumlah Keterlambatan')
ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)
