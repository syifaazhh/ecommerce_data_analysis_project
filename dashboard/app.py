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
    st.metric(label="Unique Products Sold", value=order_items_df.product_id.nunique())

col4, col5, col6 = st.columns(3)
with col4:
    st.metric(label="Total Revenue", value=f"${order_items_df['price'].sum():,.2f}")
with col5:
    st.metric(label="Average Order Value", value=f"${order_items_df.groupby('order_id')['price'].sum().mean():,.2f}")
with col6:
    st.metric(label="Total Sellers", value=sellers_df.seller_id.nunique())

col7, col8, col9 = st.columns(3)
with col7:
    st.metric(label="Total Reviews", value=order_reviews_df.shape[0])
with col8:
    st.metric(label="Average Review Score", value=f"{order_reviews_df['review_score'].mean():.2f}")
with col9:
    st.metric(label="Top Product Category", value=order_items_df['product_category_name_english'].value_counts().idxmax())

# Visualisasi
st.header("Pertanyaan Bisnis")

# P1: Jumlah Order Berdasarkan State
with st.expander("P1: Jumlah Order Berdasarkan State"):
    order_per_state = orders_df.groupby('customer_state')['order_id'].count().reset_index().rename(columns={'order_id': 'total_order'})
    order_per_state = order_per_state.sort_values(by='total_order', ascending=False).head(5)
    order_per_state['color'] = ['#5B9BD5' if x == order_per_state['total_order'].max() else '#A2C4E4' for x in order_per_state['total_order']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=order_per_state, x='customer_state', y='total_order', palette=order_per_state['color'], ax=ax)
    plt.title("Jumlah Order Berdasarkan State")
    plt.xlabel("State")
    plt.ylabel("Jumlah Order")
    st.pyplot(fig)

# P2: Pertumbuhan Order per Tahun
with st.expander("P2: Pertumbuhan Order per Tahun"):
    order_per_year = orders_df.groupby('year_of_purchase')['order_id'].count().reset_index().rename(columns={'order_id': 'total_order'})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=order_per_year, x='year_of_purchase', y='total_order', marker="o", color="#5B9BD5", ax=ax)
    plt.title("Pertumbuhan Order per Tahun")
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Order")
    st.pyplot(fig)

# P3: Customer Active vs Inactive
with st.expander("P3: Customer Active vs Inactive"):
    customer_status = customers_df.copy()
    customer_status['status'] = customer_status['customer_id'].isin(orders_df['customer_id']).map({True: 'Active', False: 'Inactive'})
    status_count = customer_status['status'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(status_count, labels=status_count.index, autopct='%1.1f%%', colors=["#5B9BD5", "#A2C4E4"])
    plt.title("Customer Active vs Inactive")
    st.pyplot(fig)

# P4: Distribusi Rating Ulasan
with st.expander("P4: Distribusi Rating Ulasan"):
    review_scores = order_reviews_df['review_score'].value_counts().reset_index().rename(columns={'index': 'review_score', 'review_score': 'total_review'})
    review_scores['color'] = ['#5B9BD5' if x == review_scores['total_review'].max() else '#A2C4E4' for x in review_scores['total_review']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=review_scores, x='review_score', y='total_review', palette=review_scores['color'], ax=ax)
    plt.title("Distribusi Rating Ulasan")
    plt.xlabel("Rating")
    plt.ylabel("Jumlah Ulasan")
    st.pyplot(fig)

# P5: Kategori Produk Terpopuler
with st.expander("P5: Kategori Produk Terpopuler"):
    popular_categories = order_items_df.groupby('product_category_name_english')['order_id'].count().reset_index().rename(columns={'order_id': 'total_order'})
    popular_categories = popular_categories.sort_values(by='total_order', ascending=False).head(5)
    popular_categories['color'] = ['#5B9BD5' if x == popular_categories['total_order'].max() else '#A2C4E4' for x in popular_categories['total_order']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=popular_categories, x='product_category_name_english', y='total_order', palette=popular_categories['color'], ax=ax)
    plt.title("Kategori Produk Terpopuler")
    plt.xlabel("Kategori Produk")
    plt.ylabel("Jumlah Order")
    st.pyplot(fig)

# P6: Metode Pembayaran Paling Sering Digunakan
with st.expander("P6: Metode Pembayaran Paling Sering Digunakan"):
    payment_methods = order_payments_df['payment_type'].value_counts().reset_index().rename(columns={'index': 'payment_type', 'payment_type': 'count'})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=payment_methods, x='payment_type', y='count', color="#5B9BD5", ax=ax)
    plt.title("Metode Pembayaran Paling Sering Digunakan")
    plt.xlabel("Metode Pembayaran")
    plt.ylabel("Jumlah Penggunaan")
    st.pyplot(fig)

# P7: Waktu Pengiriman Berdasarkan Metode Pembayaran
with st.expander("P7: Waktu Pengiriman Berdasarkan Metode Pembayaran"):
    orders_df['delivery_time'] = (orders_df['order_delivered_customer_date'] - orders_df['order_purchase_timestamp']).dt.days
    payment_delivery = pd.merge(order_payments_df, orders_df[['order_id', 'delivery_time']], on='order_id', how='left')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=payment_delivery, x='payment_type', y='delivery_time', palette=["#5B9BD5", "#A2C4E4"], ax=ax)
    plt.title("Waktu Pengiriman Berdasarkan Metode Pembayaran")
    plt.xlabel("Metode Pembayaran")
    plt.ylabel("Waktu Pengiriman (Hari)")
    st.pyplot(fig)

# P8: Perbedaan Waktu Pengiriman Berdasarkan Rating Ulasan
with st.expander("P8: Waktu Pengiriman Berdasarkan Rating Ulasan"):
    orders_reviews = pd.merge(order_reviews_df, orders_df[['order_id', 'delivery_time']], on='order_id', how='left')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=orders_reviews, x='review_score', y='delivery_time', palette=["#5B9BD5", "#A2C4E4"], ax=ax)
    plt.title("Waktu Pengiriman Berdasarkan Rating Ulasan")
    plt.xlabel("Rating")
    plt.ylabel("Waktu Pengiriman (Hari)")
    st.pyplot(fig)

# P9: Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk
with st.expander("P9: Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk"):
    category_delivery = order_items_df.groupby('product_category_name_english')['delivery_time'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=category_delivery, x='delivery_time', y='product_category_name_english', color="#5B9BD5", ax=ax)
    plt.title("Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk")
    plt.xlabel("Waktu Pengiriman (Hari)")
    plt.ylabel("Kategori Produk")
    st.pyplot(fig)

# P10: Keterlambatan Pengiriman Berdasarkan Kategori Produk
with st.expander("P10: Keterlambatan Pengiriman Berdasarkan Kategori Produk"):
    order_items_df['delivery_time'] = (
        pd.to_datetime(order_items_df['order_delivered_customer_date']) -
        pd.to_datetime(order_items_df['order_purchase_timestamp'])
    ).dt.days
    order_items_df['is_late'] = (
        pd.to_datetime(order_items_df['order_delivered_customer_date']) >
        pd.to_datetime(order_items_df['order_estimated_delivery_date'])
    )
    late_delivery_counts = order_items_df[order_items_df['is_late']].groupby('product_category_name_english').size().reset_index(name='late_count')
    late_delivery_counts = late_delivery_counts.sort_values(by='late_count', ascending=False).head(10)

    # Highlighting the highest bar
    late_delivery_counts['color'] = ['#5B9BD5' if x == late_delivery_counts['late_count'].max() else '#A2C4E4' for x in late_delivery_counts['late_count']]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=late_delivery_counts, x='late_count', y='product_category_name_english', palette=late_delivery_counts['color'], ax=ax)
    plt.title("Keterlambatan Pengiriman Berdasarkan Kategori Produk")
    plt.xlabel("Jumlah Keterlambatan")
    plt.ylabel("Kategori Produk")
    st.pyplot(fig)

# P11: Rata-Rata Rating Berdasarkan Kategori Produk
with st.expander("P11: Rata-Rata Rating Berdasarkan Kategori Produk"):
    product_ratings = pd.merge(order_items_df, order_reviews_df, on='order_id', how='left')
    avg_rating_per_category = product_ratings.groupby('product_category_name_english')['review_score'].mean().reset_index()
    avg_rating_per_category = avg_rating_per_category.sort_values(by='review_score', ascending=False).head(10)

    # Highlighting the highest bar
    avg_rating_per_category['color'] = ['#5B9BD5' if x == avg_rating_per_category['review_score'].max() else '#A2C4E4' for x in avg_rating_per_category['review_score']]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=avg_rating_per_category, x='review_score', y='product_category_name_english', palette=avg_rating_per_category['color'], ax=ax)
    plt.title("Rata-Rata Rating Berdasarkan Kategori Produk")
    plt.xlabel("Rata-Rata Rating")
    plt.ylabel("Kategori Produk")
    st.pyplot(fig)
