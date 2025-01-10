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

# Fungsi prediksi menggunakan YOLO
def prediction(image, conf):
    result = model.predict(image, conf=conf)
    res_plotted = result[0].plot()[:, :, ::-1]
    return res_plotted

# Fungsi untuk menghapus gambar
def delete_image(image_id):
    try:
        c.execute("DELETE FROM images WHERE id=?", (image_id,))
        conn.commit()
        st.experimental_rerun()
    except sqlite3.Error as e:
        st.error(f"Kesalahan saat menghapus data: {e}")

# Halaman Home
def home_page():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Selamat Datang di Aplikasi Deteksi Penyakit Daun Mangga 🌿</h1>", unsafe_allow_html=True)
    st.image(
        "https://akcdn.detik.net.id/community/media/visual/2019/11/05/962485cf-2343-402d-b80c-91d7b9199129_169.jpeg?w=620",
        caption="Aplikasi Deteksi Penyakit Daun Mangga",
        use_column_width=True
    )
    st.markdown("""
    <p style='text-align: justify;'>
    Aplikasi ini menggunakan teknologi <strong>deep learning</strong> untuk mendeteksi penyakit pada daun mangga.
    Dengan pendekatan ini, aplikasi dirancang untuk memberikan hasil deteksi yang cepat dan akurat.
    </p>
    """, unsafe_allow_html=True)

# Halaman Operasi Deteksi
def detection_page():
    st.markdown("<h1 style='text-align: center; color: #FF5722;'>Operasi Deteksi 🔍</h1>", unsafe_allow_html=True)
    confidence = st.slider('Tingkat Kepercayaan (Confidence)', 0.1, 1.0, 0.5)
    st.markdown("<p style='text-align: justify; color: #888;'>Default: 0.5. Ubah nilai ini sesuai kebutuhan.</p>", unsafe_allow_html=True)

    # Tab upload dan kamera
    tab_upload, tab_camera = st.tabs(['📂 Upload Gambar', '📷 Kamera'])

    with tab_upload:
        uploaded_image = st.file_uploader('Unggah gambar:', type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            # Simpan hasil ke database
            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('upload', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan!")

    with tab_camera:
        camera_image = st.camera_input('Ambil foto:')
        if camera_image:
            image = Image.open(camera_image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            # Simpan hasil ke database
            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('camera', img_bytes))
            conn.commit()
            st.success("Hasil deteksi berhasil disimpan!")

# Halaman Hasil Deteksi
def view_results_page():
    st.markdown("<h1 style='text-align: center; color: #FF5722;'>Hasil Deteksi 📊</h1>", unsafe_allow_html=True)
    results = c.execute("SELECT id, image FROM images ORDER BY id DESC").fetchall()
    if not results:
        st.info("Belum ada hasil deteksi.")
        return

    for image_id, img_blob in results:
        st.image(img_blob, caption=f"Hasil Deteksi #{image_id}", use_column_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download", img_blob, file_name=f"Hasil_{image_id}.png", mime="image/png")
        with col2:
            if st.button("🗑️ Hapus", key=f"delete_{image_id}"):
                delete_image(image_id)

# Sidebar navigasi
st.sidebar.button("🏠 Home", on_click=lambda: st.session_state.update({"page": "Home"}))
st.sidebar.button("🔍 Operasi Deteksi", on_click=lambda: st.session_state.update({"page": "Operasi Deteksi"}))
st.sidebar.button("📊 Hasil Deteksi", on_click=lambda: st.session_state.update({"page": "Hasil Deteksi"}))

# Halaman utama
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Operasi Deteksi":
    detection_page()
elif st.session_state.page == "Hasil Deteksi":
    view_results_page()

# Tutup koneksi database
conn.close()
