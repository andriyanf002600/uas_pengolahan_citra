import streamlit as st
from ultralytics import YOLO
from PIL import Image
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
model = YOLO("best.pt")

def prediction(image, conf):
    result = model.predict(image, conf=conf)
    res_plotted = result[0].plot()[:, :, ::-1]
    return res_plotted

def delete_image(image_id):
    c.execute("DELETE FROM images WHERE id=?", (image_id,))
    conn.commit()
    st.experimental_rerun()

# Halaman Home
def home_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Selamat Datang di Aplikasi Deteksi Penyakit Daun Mangga 🌿</h1>",
        unsafe_allow_html=True,
    )
    st.image("https://akcdn.detik.net.id/community/media/visual/2019/11/05/962485cf-2343-402d-b80c-91d7b9199129_169.jpeg?w=620", 
             caption="Aplikasi Deteksi Penyakit Daun Mangga", use_column_width=True)
    st.markdown(
        """
        <p style='text-align: justify;'>
        Aplikasi ini menggunakan teknologi <strong>efficientnet tensorflow/efficientnet Imagenet (ILSVRC-2012-CLS) classification with EfficientNet-B7</strong>
        untuk mendeteksi penyakit pada daun mangga. Dengan pendekatan deep learning, aplikasi ini dirancang untuk memberikan hasil deteksi yang cepat dan akurat.
        </p>
        """,
        unsafe_allow_html=True,
    )

# Halaman Operasi Deteksi
def detection_page():
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722;'>Operasi Deteksi 🔍</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: justify;'>Gunakan fitur ini untuk mendeteksi penyakit pada daun mangga.</p>",
        unsafe_allow_html=True,
    )
    st.file_uploader("Unggah gambar untuk dideteksi", type=["jpg", "jpeg", "png"])

# Halaman Hasil Deteksi
def view_results_page():
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722;'>Hasil Deteksi 📊</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Berikut adalah hasil deteksi yang telah Anda lakukan.</p>",
        unsafe_allow_html=True,
    )
    st.info("Belum ada hasil deteksi.", icon="ℹ️")

# CSS untuk Sidebar yang Responsif
st.sidebar.markdown(
    """
    <style>
    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 15px; /* Jarak antar tombol */
        padding: 0;
    }
    .sidebar-button {
        padding: 10px 15px; /* Padding seragam */
        margin: 0 auto; /* Tengah otomatis */
        border-radius: 8px; /* Sudut membulat */
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        color: #333; /* Warna teks */
        background-color: #f5f5f5; /* Warna latar */
        border: 1px solid #ddd; /* Border */
        width: 90%; /* Lebar proporsional */
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .sidebar-button:hover {
        background-color: #e0e0e0; /* Latar hover */
        transform: scale(1.03); /* Efek zoom ringan */
    }
    .sidebar-button:active {
        background-color: #ccc; /* Latar klik */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Navigasi Sidebar
st.sidebar.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
st.sidebar.markdown("<div class='sidebar-button'>🔍 Operasi Deteksi</div>", unsafe_allow_html=True)
if st.sidebar.button("📊 Hasil Deteksi"):
    st.session_state.page = "Hasil Deteksi"

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Pilih halaman berdasarkan navigasi
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
