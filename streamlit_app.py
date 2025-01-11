import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Inisialisasi database
conn = sqlite3.connect('database.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS images
             (id INTEGER PRIMARY KEY AUTOINCREMENT, 
              tab TEXT, 
              image BLOB)''')
conn.commit()

# Inisialisasi dua model YOLO
model_top = YOLO("best.pt")   # Model untuk tab atas
model_bottom = YOLO("best1.pt")  # Model untuk tab bawah

def prediction(image, conf, model):
    result = model.predict(image, conf=conf)
    res_plotted = result[0].plot()[:, :, ::-1]
    return res_plotted

def delete_image(image_id):
    c.execute("DELETE FROM images WHERE id=?", (image_id,))
    conn.commit()
    st.experimental_rerun()

# Halaman Home (Kosong)
def home_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Selamat Datang di Aplikasi Deteksi Penyakit Daun Mangga 🌿</h1>",
        unsafe_allow_html=True,
    )
    st.image("https://akcdn.detik.net.id/community/media/visual/2019/11/05/962485cf-2343-402d-b80c-91d7b9199129_169.jpeg?w=620", caption="Aplikasi Deteksi Penyakit Daun Mangga", use_column_width=True)
    st.markdown(
        """
        <p style='text-align: justify;'>
        Aplikasi ini menggunakan dua teknologi yang bisa dipilih <strong>YOLO v8 Dan EfficientNet-B7</strong> untuk mendeteksi penyakit pada daun mangga.
        Dengan pendekatan deep learning, aplikasi ini dirancang untuk memberikan hasil deteksi yang cepat dan akurat.
        </p>
        """,
        unsafe_allow_html=True,
    )

# Inisialisasi model YOLO
model1 = YOLO("best.pt")  # Model pertama
model2 = YOLO("best1.pt")  # Model kedua

def prediction_with_model(image, conf, model):
    result = model.predict(image, conf=conf)
    res_plotted = result[0].plot()[:, :, ::-1]
    return res_plotted

# Halaman Operasi Deteksi
def detection_page():
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722;'>Operasi Deteksi 🔍</h1>",
        unsafe_allow_html=True,
    )
    confidence = st.slider('Pilih Tingkat Kepercayaan (Confidence)', 0.1, 1.0, 0.5)
    st.markdown(
        """
        <p style='text-align: justify; font-size: 14px; color: #888;'>
        0.50 adalah default. Jika diubah kurang atau lebih, kemungkinan hasil tidak sesuai atau terjadi error.
        </p>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(['📂 Model YOLOv8: Upload Gambar', '📂 Model EfficientNet-B7: Upload Gambar'])  # Tab untuk kedua model

    with tab1:
        st.markdown("<h3>Unggah Gambar (Model: YOLOv8)</h3>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader('Pilih gambar dari perangkat Anda (Model YOLOv8):', type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            pred = prediction_with_model(image, confidence, model1)
            st.image(pred, caption="Hasil Deteksi (Model 1)", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('model1_upload', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan dengan Model YOLOv8!", icon="✅")

    with tab2:
        st.markdown("<h3>Unggah Gambar (Model: EfficientNet-B7)</h3>", unsafe_allow_html=True)
        uploaded_image_model2 = st.file_uploader('Pilih gambar dari perangkat Anda (Model EfficientNet-B7):', type=['jpg', 'jpeg', 'png'])
        if uploaded_image_model2:
            image = Image.open(uploaded_image_model2)
            pred = prediction_with_model(image, confidence, model2)
            st.image(pred, caption="Hasil Deteksi (Model EfficientNet-B7)", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('model2_upload', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan dengan Model 2!", icon="✅")

# Halaman Hasil Deteksi
def view_results_page():
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722;'>Hasil Deteksi 📊</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='text-align: center;'>Berikut adalah daftar hasil deteksi yang telah Anda lakukan.</p>", unsafe_allow_html=True)

    images = c.execute("SELECT id, image FROM images ORDER BY id DESC").fetchall()
    if not images:
        st.info("Belum ada hasil deteksi.", icon="ℹ️")
        return

    for image_id, img in images:
        st.image(img, caption=f"Hasil Deteksi #{image_id}", use_column_width=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "⬇️ Download Hasil",
                img,
                file_name=f"Deteksi_Penyakit_{image_id}.png",
                mime="image/png",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Hapus", key=f"delete_{image_id}"):
                delete_image(image_id)
                st.success("Gambar berhasil dihapus!", icon="✅")

# Halaman Statistik
def statistics_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Statistik 📈</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: justify;'>Halaman ini menampilkan statistik dari hasil deteksi yang telah dilakukan.</p>",
        unsafe_allow_html=True,
    )
    
    data = c.execute("SELECT tab, COUNT(*) as count FROM images GROUP BY tab").fetchall()
    if not data:
        st.info("Belum ada data untuk ditampilkan.", icon="ℹ️")
        return

    df = pd.DataFrame(data, columns=['Model', 'Jumlah Deteksi'])
    st.dataframe(df)

    st.markdown("<h3>Grafik Jumlah Deteksi per Model</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.bar(df['Model'], df['Jumlah Deteksi'], color=['#FF5722', '#4CAF50'])
    ax.set_title("Jumlah Deteksi per Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Jumlah Deteksi")
    st.pyplot(fig)

    st.markdown("<h3>Visualisasi Interaktif</h3>", unsafe_allow_html=True)
    fig_plotly = px.bar(
        df, 
        x='Model', 
        y='Jumlah Deteksi', 
        color='Model',
        title="Jumlah Deteksi per Model (Interaktif)",
        text='Jumlah Deteksi'
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    st.markdown("<h3>Foto Grafik Contoh</h3>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bar-chart-example.svg/1920px-Bar-chart-example.svg.png", caption="Contoh Grafik Bar", use_column_width=True)

# Sidebar Navigasi
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Main Menu</h2>", unsafe_allow_html=True)
st.sidebar.button("🏠 Home", on_click=lambda: st.session_state.update(page="Home"))
st.sidebar.button("🔍 Operasi Deteksi", on_click=lambda: st.session_state.update(page="Operasi Deteksi"))
st.sidebar.button("📊 Hasil Deteksi", on_click=lambda: st.session_state.update(page="Hasil Deteksi"))
st.sidebar.button("📈 Statistik", on_click=lambda: st.session_state.update(page="Statistik"))

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Operasi Deteksi":
    detection_page()
elif st.session_state.page == "Hasil Deteksi":
    view_results_page()
elif st.session_state.page == "Statistik":
    statistics_page()

st.markdown("<hr><p style='text-align: center;'>© Andriyan Firmansyah-227006416022-Pengolahan Citra</p>", unsafe_allow_html=True)
conn.close()
