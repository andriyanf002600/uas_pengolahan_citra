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

# Halaman Home (Informasi)
def home_page():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Aplikasi Deteksi Penyakit Pada Daun Mangga 🌿</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='text-align: justify; font-size: 18px;'>
        Aplikasi ini dirancang untuk mendeteksi penyakit pada daun mangga menggunakan teknologi terkini. 
        Dengan menggunakan algoritma deteksi berbasis YOLO (You Only Look Once), 
        aplikasi ini mampu mengenali pola dan gejala penyakit pada gambar daun mangga secara akurat.
        </p>
        <h3>Arsitektur Aplikasi</h3>
        <ul style='font-size: 18px;'>
            <li><strong>YOLO:</strong> Model deteksi objek yang cepat dan akurat untuk menganalisis gambar daun mangga.</li>
            <li><strong>Streamlit:</strong> Framework Python untuk membuat antarmuka aplikasi web yang interaktif.</li>
            <li><strong>SQLite:</strong> Database ringan untuk menyimpan hasil deteksi.</li>
        </ul>
        <p style='text-align: justify; font-size: 18px;'>
        Aplikasi ini juga mendukung pengambilan foto langsung melalui kamera atau unggah gambar dari perangkat.
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
    confidence = st.slider('Pilih Tingkat Kepercayaan (Confidence)', 0.1, 1.0, 0.5)
    st.markdown(
        """
        <p style='text-align: justify; font-size: 14px; color: #888;'>
        0.5 adalah default. Jika diubah kurang atau lebih, kemungkinan hasil tidak sesuai atau terjadi error.
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
        "<h1 style='text-align: center; color: #FF5722;'>Hasil Deteksi 🔍</h1>",
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

# Navigasi Sidebar yang Tidak Menyerupai Radio Button
st.sidebar.title("🔄 Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Halaman",
    ["🏠 Home", "Operasi Deteksi", "Hasil Deteksi"],
    format_func=lambda x: x.split(" ")[1],  # Hanya menampilkan teks tanpa ikon
)

if menu == "Home":
    home_page()
elif menu == "Operasi Deteksi":
    detection_page()
elif menu == "Hasil Deteksi":
    view_results_page()

# Tutup koneksi database
conn.close()
