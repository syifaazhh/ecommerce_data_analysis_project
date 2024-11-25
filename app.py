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

# KPI Section
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Total Customers", value=customers_df.customer_id.nunique())
with col2:
    st.metric(label="Total Orders", value=orders_df.shape[0])

# Visualisasi
st.header("Pertanyaan Bisnis")

# P1: Jumlah Order Berdasarkan State
with st.expander("P1: Jumlah Order Berdasarkan State"):
    tab1, tab2 = st.tabs(['Terbanyak', 'Tersedikit'])
    with tab1:
        top_state_df = order_per_state_df.sort_values(by='total_order', ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_state_df, x='state', y='total_order', ax=ax, palette="Blues_d")
        plt.title("Top States Berdasarkan Order")
        st.pyplot(fig)
    with tab2:
        worst_state_df = order_per_state_df.sort_values(by='total_order').head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=worst_state_df, x='state', y='total_order', ax=ax, palette="Reds_d")
        plt.title("State dengan Order Tersedikit")
        st.pyplot(fig)

# P2: Pertumbuhan Order per Tahun
with st.expander("P2: Pertumbuhan Order per Tahun"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=order_per_year_df, x='year_of_purchase', y='total_order', marker="o", ax=ax)
    plt.title("Jumlah Order per Tahun")
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Order")
    st.pyplot(fig)

# P3: Customer Active vs Inactive
with st.expander("P3: Customer Active vs Inactive"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(customer_status_df['total_customer'], labels=customer_status_df['status'], autopct='%1.1f%%', explode=(0.1, 0), colors=["#72BCD4", "#D3D3D3"])
    plt.title("Rasio Customer Active vs Inactive")
    st.pyplot(fig)

# P4: Distribusi Rating Ulasan
with st.expander("P4: Distribusi Rating Ulasan"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=review_score_df, x='review_score', y='total_review', palette="RdYlGn", ax=ax)
    plt.title("Distribusi Rating Ulasan")
    plt.xlabel("Rating")
    plt.ylabel("Total Review")
    st.pyplot(fig)

# P5: Kategori Produk Terpopuler
with st.expander("P5: Kategori Produk Terpopuler"):
    best_category_df = order_product_category_df.sort_values(by='total_order', ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=best_category_df, x='product_category_name_english', y='total_order', palette="coolwarm", ax=ax)
    plt.title("Top 5 Kategori Produk")
    plt.xlabel("Kategori Produk")
    plt.ylabel("Jumlah Order")
    st.pyplot(fig)

# P6: Metode Pembayaran Paling Sering Digunakan
with st.expander("P6: Metode Pembayaran Paling Sering Digunakan"):
    payment_methods = order_payments_df['payment_type'].value_counts().reset_index()
    payment_methods.columns = ['payment_type', 'count']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=payment_methods, x='count', y='payment_type', palette="Purples_d", ax=ax)
    plt.title("Distribusi Metode Pembayaran")
    plt.xlabel("Jumlah")
    plt.ylabel("Metode Pembayaran")
    st.pyplot(fig)

# P7: Waktu Pengiriman Berdasarkan Metode Pembayaran
with st.expander("P7: Hubungan Metode Pembayaran dan Waktu Pengiriman"):
    orders_payments = pd.merge(order_payments_df, orders_df, on='order_id', how='left')
    orders_payments['delivery_time'] = (orders_payments['order_delivered_customer_date'] - orders_payments['order_purchase_timestamp']).dt.days
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=orders_payments, x='payment_type', y='delivery_time', palette="coolwarm", ax=ax)
    plt.title("Waktu Pengiriman Berdasarkan Metode Pembayaran")
    plt.xlabel("Metode Pembayaran")
    plt.ylabel("Waktu Pengiriman (Hari)")
    st.pyplot(fig)

# P8: Distribusi Rating Ulasan
with st.expander("P8: Distribusi Rating Ulasan"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=review_score_df, x='review_score', y='total_review', palette="RdYlGn", ax=ax)
    plt.title("Distribusi Rating Ulasan")
    plt.xlabel("Rating")
    plt.ylabel("Jumlah Ulasan")
    st.pyplot(fig)

# P9: Perbedaan Waktu Pengiriman Berdasarkan Ulasan
with st.expander("P9: Waktu Pengiriman Berdasarkan Ulasan Positif vs Negatif"):
    orders_reviews = pd.merge(order_reviews_df, orders_df, on='order_id', how='left')
    orders_reviews['delivery_time'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_purchase_timestamp']).dt.days
    orders_reviews['sentiment'] = orders_reviews['review_score'].apply(lambda x: "Positive" if x >= 4 else "Negative")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=orders_reviews, x='sentiment', y='delivery_time', palette="coolwarm", ax=ax)
    plt.title("Waktu Pengiriman Berdasarkan Sentimen Ulasan")
    plt.xlabel("Sentimen Ulasan")
    plt.ylabel("Waktu Pengiriman (Hari)")
    st.pyplot(fig)

# P10: Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk
with st.expander("P10: Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk"):
    order_items_product_df['delivery_time'] = (
        pd.to_datetime(order_items_product_df['order_delivered_customer_date']) -
        pd.to_datetime(order_items_product_df['order_purchase_timestamp'])
    ).dt.days
    avg_delivery = order_items_product_df.groupby('product_category_name_english')['delivery_time'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(y=avg_delivery.index, x=avg_delivery.values, palette="coolwarm", ax=ax)
    plt.title("Rata-Rata Waktu Pengiriman Berdasarkan Kategori Produk")
    plt.xlabel("Waktu Pengiriman (Hari)")
    plt.ylabel("Kategori Produk")
    st.pyplot(fig)

# P11: Keterlambatan Pengiriman Berdasarkan Kategori Produk
with st.expander("P11: Keterlambatan Pengiriman Berdasarkan Kategori Produk"):
    order_items_product_df['is_late'] = order_items_product_df['delivery_time'] > (
        pd.to_datetime(order_items_product_df['order_estimated_delivery_date']) -
        pd.to_datetime(order_items_product_df['order_purchase_timestamp'])
    ).dt.days
    late_delivery = order_items_product_df[order_items_product_df['is_late']].groupby('product_category_name_english').size().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(y=late_delivery.index, x=late_delivery.values, palette="Reds_d", ax=ax)
    plt.title("Keterlambatan Pengiriman Berdasarkan Kategori Produk")
    plt.xlabel("Jumlah Keterlambatan")
    plt.ylabel("Kategori Produk")
    st.pyplot(fig)