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

# Halaman utama
def main_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>Deteksi Penyakit Pada Daun Mangga 🌿</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 18px;'>Gunakan aplikasi ini untuk mendeteksi penyakit pada daun mangga menggunakan kamera atau gambar yang diunggah.</p>",
        unsafe_allow_html=True,
    )

    confidence = st.slider('Pilih Tingkat Kepercayaan (Confidence)', 0.1, 1.0, 0.5, help="Pilih tingkat kepercayaan deteksi penyakit pada daun")
    st.markdown(f"<p style='text-align: center; font-size: 16px;'>Confidence: <strong>{confidence}</strong></p>", unsafe_allow_html=True)

    tab2, tab1 = st.tabs(['📂 Upload Gambar', '📷 Kamera'])  # Upload Gambar di kiri, Kamera di kanan

    with tab2:  # Upload Gambar
        st.markdown("<h3 style='font-family: Arial, sans-serif;'>Unggah Gambar</h3>", unsafe_allow_html=True)
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
        st.markdown("<h3 style='font-family: Arial, sans-serif;'>Ambil Foto dengan Kamera</h3>", unsafe_allow_html=True)
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

# Halaman hasil deteksi
def view_results_page():
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722; font-family: Arial, sans-serif;'>Hasil Deteksi 🔍</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='text-align: center; font-size: 18px;'>Berikut adalah daftar hasil deteksi yang telah Anda lakukan.</p>", unsafe_allow_html=True)

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

# Navigasi kotak di sidebar
st.sidebar.title("🔄 Navigasi")
st.sidebar.markdown(
    """
    <style>
    .nav-button {
        display: block;
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .nav-button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Default ke Home saat pertama kali masuk
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "📊 Hasil Deteksi"], index=0)

if menu == "🏠 Home":
    main_page()
elif menu == "📊 Hasil Deteksi":
    view_results_page()

# Tutup koneksi database
conn.close()
