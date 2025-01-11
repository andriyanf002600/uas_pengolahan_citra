import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import sqlite3

# Inisialisasi database
conn = sqlite3.connect('database.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS images
             (id INTEGER PRIMARY KEY AUTOINCREMENT, 
              tab TEXT, 
              image BLOB)''')
conn.commit()

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

    tab1, tab2 = st.tabs(['📂 Model 1: Upload Gambar', '📂 Model 2: Upload Gambar'])  # Tab untuk kedua model

    with tab1:  # Upload Gambar dengan Model 1
        st.markdown("<h3>Unggah Gambar (Model: best.pt)</h3>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader('Pilih gambar dari perangkat Anda (Model 1):', type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            pred = prediction_with_model(image, confidence, model1)
            st.image(pred, caption="Hasil Deteksi (Model 1)", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('model1_upload', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan dengan Model 1!", icon="✅")

    with tab2:  # Upload Gambar dengan Model 2
        st.markdown("<h3>Unggah Gambar (Model: best1.pt)</h3>", unsafe_allow_html=True)
        uploaded_image_model2 = st.file_uploader('Pilih gambar dari perangkat Anda (Model 2):', type=['jpg', 'jpeg', 'png'])
        if uploaded_image_model2:
            image = Image.open(uploaded_image_model2)
            pred = prediction_with_model(image, confidence, model2)
            st.image(pred, caption="Hasil Deteksi (Model 2)", use_column_width=True)

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

# Navigasi Sidebar
st.sidebar.markdown(
    """
    <style>
    .sidebar-box {
        display: block; /* Mengatur elemen menjadi block */
        padding: 15px; /* Jarak dalam elemen */
        margin: 10px 0; /* Jarak antar elemen */
        border-radius: 10px; /* Sudut membulat */
        text-align: center; /* Teks rata tengah */
        font-weight: bold; /* Teks tebal */
        cursor: pointer; /* Menunjukkan elemen bisa diklik */
        background-color: #f0f0f0; /* Warna latar belakang */
        border: 1px solid #ccc; /* Border dengan warna abu-abu */
        width: 100%; /* Lebar penuh */
    }
    .sidebar-box:hover {
        background-color: #e0e0e0; /* Warna latar belakang saat hover */
        border-color: #bbb; /* Warna border saat hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Fungsi Navigasi
def navigate_to(page):
    st.session_state.page = page

# Tombol Navigasi
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Main Menu</h2>", unsafe_allow_html=True)
st.sidebar.button("🏠 Home", on_click=navigate_to, args=("Home",), key="home_btn", help="Kembali ke halaman Home")
st.sidebar.button("🔍 Operasi Deteksi", on_click=navigate_to, args=("Operasi Deteksi",), key="detect_btn", help="Pergi ke Operasi Deteksi")
st.sidebar.button("📊 Hasil Deteksi", on_click=navigate_to, args=("Hasil Deteksi",), key="results_btn", help="Lihat hasil deteksi")

# Halaman berdasarkan navigasi
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Operasi Deteksi":
    detection_page()
elif st.session_state.page == "Hasil Deteksi":
    view_results_page()

# Footer Copyright di semua halaman
st.markdown(
    "<hr><p style='text-align: center;'>© Andriyan Firmansyah-227006416022-Pengolahan Citra</p>",
    unsafe_allow_html=True,
)

# Tutup koneksi database
conn.close()
