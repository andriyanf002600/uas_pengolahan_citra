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
    st.title('Deteksi Penyakit Pada Daun Mangga')
    confidence = st.slider('Pilih Confidence', 0.1, 1.0, 0.5)
    st.write(f'Confidence: {confidence}')

    tab1, tab2 = st.tabs(['📷 Kamera', '📂 Upload'])

    with tab1:
        image = st.camera_input('Ambil Foto')
        if image:
            image = Image.open(image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('camera', img_bytes))
            conn.commit()

    with tab2:
        uploaded_image = st.file_uploader('Upload Gambar', type=['jpg', 'jpeg', 'png'])
        if uploaded_image:
            image = Image.open(uploaded_image)
            pred = prediction(image, confidence)
            st.image(pred, caption="Hasil Deteksi", use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(pred).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            c.execute("INSERT INTO images (tab, image) VALUES (?, ?)", ('upload', img_bytes))
            conn.commit()

# Halaman hasil deteksi
def view_results_page():
    st.title('Hasil Deteksi')
    images = c.execute("SELECT id, image FROM images ORDER BY id DESC").fetchall()
    if not images:
        st.info("Belum ada hasil deteksi.")
        return

    for image_id, img in images:
        st.image(img, caption=f'Hasil Deteksi #{image_id}')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button("Download", img, file_name=f"Deteksi_Penyakit_{image_id}.png", mime="image/png")
        with col2:
            if st.button("Hapus", key=f"delete_{image_id}"):
                delete_image(image_id)

# Navigasi sidebar
st.sidebar.title('Navigasi')
menu = st.sidebar.radio('Pilih Halaman', ['Home', 'Hasil Deteksi'])

if menu == 'Home':
    main_page()
elif menu == 'Hasil Deteksi':
    view_results_page()

# Tutup koneksi database
conn.close()
