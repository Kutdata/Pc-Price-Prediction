# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:58:09 2025

@author: MUSTAFA
"""
import streamlit as st
import pandas as pd
import pickle

# Modelleri yükle
lnr_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgboost_model.pkl', 'rb'))
lgb_model = pickle.load(open('lightgbm_model.pkl', 'rb'))

# Veri setini yükle (özellikleri almak için)
df = pd.read_csv('laptop_prices.csv')

# Streamlit arayüzü
st.title('Dizüstü Bilgisayar Fiyat Tahmini')

# Kullanıcıdan girdi al
screen_size = st.number_input('Ekran Boyutu (inç)', min_value=10.0, max_value=20.0, value=15.6)
ram = st.number_input('RAM (GB)', min_value=4, max_value=64, value=8)
storage = st.number_input('Depolama (GB)', min_value=128, max_value=2048, value=512)
ppi = st.number_input('PPI', min_value=100, max_value=400, value=220)
weight = st.number_input('Ağırlık (kg)', min_value=1.0, max_value=5.0, value=2.0)
# Kategorik özellikler için seçim kutuları
brand = st.selectbox('Marka', df['Brand'].unique())
processor = st.selectbox('İşlemci', df['Processor'].unique())
gpu = st.selectbox('GPU', df['GPU'].unique())
os = st.selectbox('İşletim Sistemi', df['Operating System'].unique())

# Tahmin yap
if st.button('Tahmin Et'):
    # Kullanıcı girdilerini DataFrame'e dönüştür
    user_data = pd.DataFrame({
        'Screen Size (inch)': [screen_size],
        'RAM (GB)': [ram],
        'Storage': [storage],
        'PPI': [ppi],
        'Weight (kg)': [weight],
        'Brand': [brand],
        'Processor': [processor],
        'GPU': [gpu],
        'Operating System': [os]
    })

    # Kategorik özellikleri one-hot encode et
    user_data = pd.get_dummies(user_data, columns=['Brand', 'Processor', 'GPU', 'Operating System'], drop_first=True)

    # Eğitim verisindeki sütunlara sahip olduğundan emin ol
    train_cols = lnr_model.feature_names_in_
    for col in train_cols:
        if col not in user_data.columns:
            user_data[col] = 0
    user_data = user_data[train_cols]

    # Tahminleri yap
    lnr_prediction = lnr_model.predict(user_data)[0]
    xgb_prediction = xgb_model.predict(user_data)[0]
    lgb_prediction = lgb_model.predict(user_data)[0]

    # Tahminleri göster
    st.subheader('Tahminler:')
    st.write(f'Linear Regression: ${lnr_prediction:.2f}')
    st.write(f'XGBoost: ${xgb_prediction:.2f}')
    st.write(f'LightGBM: ${lgb_prediction:.2f}')
