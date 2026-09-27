"""
Prediksi Harga Laptop - Streamlit App
Menggunakan model Linear Regression yang sudah dilatih & disimpan dari
notebook 'Laptop_Price_Preprocessing_Training.ipynb' (Google Colab).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

MODEL_PATH = "laptop_price_model.pkl"
METADATA_PATH = "model_metadata.json"

# Kurs EUR -> IDR. Model tetap dilatih & memprediksi dalam Euro;
DEFAULT_EUR_TO_IDR = 20404


@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


def build_feature_vector(inputs, feature_columns):
    """Susun 1 baris DataFrame dengan urutan kolom PERSIS seperti saat training."""
    row = pd.Series(0, index=feature_columns, dtype=float)

    # fitur numerik langsung
    numeric_map = {
        "Inches": inputs["inches"],
        "Ram": inputs["ram"],
        "Weight": inputs["weight"],
        "Cpu_Speed_GHz": inputs["cpu_speed"],
        "Storage_GB": inputs["storage_gb"],
        "PPI": inputs["ppi"],
        "Touchscreen": int(inputs["touchscreen"]),
        "IPS_Panel": int(inputs["ips_panel"]),
    }
    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = val

    # fitur kategorikal (one-hot): set kolom yang sesuai jadi 1
    onehot_map = {
        "Company": inputs["company"],
        "TypeName": inputs["type_name"],
        "OS": inputs["os"],
        "Cpu_Brand": inputs["cpu_brand"],
        "Gpu_Brand": inputs["gpu_brand"],
        "Storage_Type": inputs["storage_type"],
    }
    for prefix, value in onehot_map.items():
        col_name = f"{prefix}_{value}"
        if col_name in row.index:
            row[col_name] = 1

    return row.values.reshape(1, -1)


def main():
    st.set_page_config(page_title="Prediksi Harga Laptop", page_icon="💻", layout="centered")

    st.markdown(
        """
        <div style="background-color:#726E6D;padding:10px;border-radius:6px;">
        <h3 style="color:white;margin:0;">Prediksi Harga Laptop Menggunakan Regresi Linier</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)):
        st.error(
            "File model belum ditemukan. Pastikan 'laptop_price_model.pkl' dan "
            "'model_metadata.json' (hasil download dari notebook Colab) ada satu folder "
            "dengan app.py ini."
        )
        return

    model, metadata = load_model_and_metadata()
    feature_columns = metadata["feature_columns"]
    options = metadata["category_options"]
    ranges = metadata["numeric_ranges"]

    menu = st.sidebar.selectbox("Menu", ["Prediksi Harga Laptop", "Tentang Aplikasi"])

    if menu == "Tentang Aplikasi":
        st.subheader("Tentang Aplikasi")
        st.write(
            "Aplikasi ini memprediksi harga laptop (dalam Euro) berdasarkan spesifikasi "
            "menggunakan model Linear Regression yang dilatih dari dataset laptop_price.csv. "
            "Data sudah melalui tahap preprocessing (ekstraksi RAM, penyimpanan, CPU, GPU, "
            "resolusi layar, dsb.) sebelum digunakan untuk melatih model di Google Colab."
        )
        st.write(f"Jumlah fitur yang digunakan model: **{len(feature_columns)}**")
        return

    st.subheader("Masukkan Spesifikasi Laptop")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Merek (Company)", options["Company"])
        type_name = st.selectbox("Tipe Laptop", options["TypeName"])
        os_choice = st.selectbox("Sistem Operasi", options["OS"])
        cpu_brand = st.selectbox("Merek CPU", options["Cpu_Brand"])
        gpu_brand = st.selectbox("Merek GPU", options["Gpu_Brand"])
        storage_type = st.selectbox("Tipe Penyimpanan", options["Storage_Type"])

    with col2:
        inches = st.slider("Ukuran Layar (inch)", float(ranges["Inches"][0]), float(ranges["Inches"][1]), float(ranges["Inches"][0]))
        ram = st.slider("RAM (GB)", int(ranges["Ram"][0]), int(ranges["Ram"][1]), int(ranges["Ram"][0]), step=2)
        weight = st.slider("Berat (kg)", float(ranges["Weight"][0]), float(ranges["Weight"][1]), float(ranges["Weight"][0]))
        cpu_speed = st.slider("Kecepatan CPU (GHz)", float(ranges["Cpu_Speed_GHz"][0]), float(ranges["Cpu_Speed_GHz"][1]), float(ranges["Cpu_Speed_GHz"][0]))
        storage_gb = st.slider("Kapasitas Penyimpanan (GB)", float(ranges["Storage_GB"][0]), float(ranges["Storage_GB"][1]), float(ranges["Storage_GB"][0]))

    st.markdown("**Resolusi Layar**")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        res_width = st.number_input("Lebar (pixel)", min_value=800, max_value=4000, value=1920, step=10)
    with res_col2:
        res_height = st.number_input("Tinggi (pixel)", min_value=600, max_value=3000, value=1080, step=10)

    ppi = ((res_width ** 2 + res_height ** 2) ** 0.5) / inches if inches else 0.0
    st.caption(f"PPI (kepadatan pixel) terhitung: {ppi:.1f}")

    touchscreen = st.checkbox("Layar Sentuh (Touchscreen)")
    ips_panel = st.checkbox("Panel IPS")

    st.markdown("**Mata Uang Tampilan**")
    kurs = st.number_input(
        "Kurs 1 EUR ke IDR",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_EUR_TO_IDR,
        step=50,
        help="Model dilatih dan memprediksi dalam Euro; angka ini hanya untuk konversi tampilan ke Rupiah.",
    )

    if st.button("Prediksi Harga"):
        inputs = {
            "company": company,
            "type_name": type_name,
            "os": os_choice,
            "cpu_brand": cpu_brand,
            "gpu_brand": gpu_brand,
            "storage_type": storage_type,
            "inches": inches,
            "ram": ram,
            "weight": weight,
            "cpu_speed": cpu_speed,
            "storage_gb": storage_gb,
            "ppi": ppi,
            "touchscreen": touchscreen,
            "ips_panel": ips_panel,
        }

        X = build_feature_vector(inputs, feature_columns)
        predicted_price_eur = model.predict(X)[0]
        predicted_price_eur = max(predicted_price_eur, 0)
        predicted_price_idr = predicted_price_eur * kurs

        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"Rp {predicted_price_idr:,.0f}".replace(",", "."))
        with col_b:
            st.info(f"€ {predicted_price_eur:,.2f}")
        st.caption("Estimasi berdasarkan model Linear Regression. Bisa meleset dari harga pasar sebenarnya.")


if __name__ == "__main__":
    main()