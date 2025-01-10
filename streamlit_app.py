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
model = YOLO("best.pt")

def prediction(image, conf):
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
        Aplikasi ini menggunakan teknologi <strong>efficientnet tensorflow/efficientnet Imagenet (ILSVRC-2012-CLS) classification with EfficientNet-B7.</strong> untuk mendeteksi penyakit pada daun mangga.
        Dengan pendekatan deep learning, aplikasi ini dirancang untuk memberikan hasil deteksi yang cepat dan akurat.
        </p>
        """,
        unsafe_allow_html=True,
    )
    # Tambahkan gambar di bawah tulisan
  

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

    tab2, tab1 = st.tabs(['📂 Upload Gambar', '📷 Kamera'])  # Upload di kiri, Kamera di kanan

    with tab2:  # Upload Gambar
        st.markdown("<h3>Unggah Gambar</h3>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader('Pilih gambar dari perangkat Anda:', type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('upload', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan!", icon="✅")

    with tab1:  # Kamera
        st.markdown("<h3>Ambil Foto dengan Kamera</h3>", unsafe_allow_html=True)
        image = st.camera_input('Klik tombol di bawah untuk mengambil foto:')
        if image:
            image = Image.open(image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('camera', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan!", icon="✅")

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
        padding: 50px; /* Jarak dalam elemen */
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
