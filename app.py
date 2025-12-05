import os
import logging
from flask import Flask, render_template, request
from joblib import load
import pandas as pd

# ---------- konfigurasi logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- app ----------
app = Flask(__name__)

# ---------- load model (sekali saat startup) ----------
MODEL_PATH = "model_kelulusan.joblib"
model = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = load(MODEL_PATH)
        logger.info("Model berhasil dimuat dari: %s", MODEL_PATH)
        # log info model kelas jika ada
        if hasattr(model, "classes_"):
            logger.info("Model classes_: %s", getattr(model, "classes_"))
        logger.info("Model punya predict_proba: %s", hasattr(model, "predict_proba"))
    except Exception as e:
        model_load_error = f"Error loading model: {e}"
        logger.exception("Gagal memuat model")
else:
    model_load_error = f"Model file not found at '{MODEL_PATH}'"
    logger.error(model_load_error)

# ---------- helper konversi & validasi ----------
def to_float(value, name):
    try:
        return float(value)
    except Exception:
        raise ValueError(f"Field '{name}' harus berupa angka (contoh: 3.25).")

def to_int(value, name):
    try:
        # terima input numeric seperti "85.0" atau "85"
        return int(float(value))
    except Exception:
        raise ValueError(f"Field '{name}' harus berupa bilangan bulat (contoh: 90).")

# ---------- SOFTBOOST: menaikkan probabilitas secara wajar ----------
def adjust_prob_softboost(prob, input_row):
    """
    Memberikan dorongan probabilitas berdasarkan kondisi fitur yang sangat baik.
    Hasil akhir tetap terlihat natural (tidak tiba-tiba 99%).
    """
    ipk = float(input_row.get('ipk', 0))
    pres = float(input_row.get('presensi', 0))
    mengulang = int(input_row.get('mengulang', 0))
    sks = float(input_row.get('sks_lulus', 0))

    boost = 0.0

    # IPK
    if ipk >= 3.9:
        boost += 0.12     # +12%
    elif ipk >= 3.7:
        boost += 0.07     # +7%

    # Presensi
    if pres >= 98:
        boost += 0.10     # +10%
    elif pres >= 95:
        boost += 0.05     # +5%

    # Tidak mengulang
    if mengulang == 0:
        boost += 0.05     # +5%

    # SKS besar → hampir selesai
    if sks >= 140:
        boost += 0.03     # +3%

    # Total boost tidak boleh kelewat ekstrem
    boosted = prob + boost

    # Maksimal ditampilkan 0.985 (98.5%) → natural, tidak 100%
    if boosted > 0.985:
        boosted = 0.985

    return boosted

# ---------- fungsi prediksi ----------
def prediksi_kelulusan(ipk, sks_lulus, presensi, mengulang):
    """
    Mengembalikan tuple: (label, prob_percent, rekomendasi)
    prob_percent: float antara 0..100
    """
    if model is None:
        raise RuntimeError(model_load_error or "Model tidak tersedia.")

    # Buat DataFrame input (urutkan kolom seperti saat training)
    df_input = pd.DataFrame([{
        "ipk": ipk,
        "sks_lulus": sks_lulus,
        "presensi": presensi,
        "mengulang": mengulang,
    }])

    # Prediksi kelas (0/1)
    y_pred = model.predict(df_input)[0]

    # Ambil probabilitas kelas positif (robust terhadap urutan classes_)
    prob_lulus = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_input)[0]
        # Jika model expose classes_, cari index label 1
        if hasattr(model, "classes_"):
            try:
                classes = list(model.classes_)
                # dukung 1 dan '1' serta 1.0
                if 1 in classes:
                    pos_index = classes.index(1)
                elif 1.0 in classes:
                    pos_index = classes.index(1.0)
                elif "1" in classes:
                    pos_index = classes.index("1")
                else:
                    # fallback: ambil indeks 1 jika tersedia, else 0
                    pos_index = 1 if len(probs) > 1 else 0
            except Exception:
                pos_index = 1 if len(probs) > 1 else 0
        else:
            pos_index = 1 if len(probs) > 1 else 0

        # ambil probabilitas pada index yang ditentukan
        try:
            prob_lulus = float(probs[pos_index])
        except Exception:
            # jika ada error, fallback ke deterministik
            prob_lulus = float(bool(y_pred))
    else:
        # model tidak punya predict_proba -> gunakan prediksi deterministik
        prob_lulus = float(bool(y_pred))

    # Pastikan berada di range 0..1
    prob_lulus = max(0.0, min(prob_lulus, 1.0))

    # --- APPLY SOFTBOOST: siapkan input_row dan terapkan adjust_prob_softboost ---
    input_row = {
        "ipk": ipk,
        "presensi": presensi,
        "sks_lulus": sks_lulus,
        "mengulang": mengulang
    }
    prob_lulus = adjust_prob_softboost(prob_lulus, input_row)
    # ---------------------------------------------------------------------------

    prob_percent = prob_lulus * 100.0

    # Label teks
    label = "Lulus tepat waktu" if (y_pred == 1 or y_pred == "1" or y_pred == 1.0) else "Berisiko terlambat"

    # Rekomendasi berdasarkan probabilitas
    if prob_percent >= 85:
        rekomendasi = "Pertahankan performa. Indeks Prestasi Semester (IPS), presensi, dan konsistensi belajar."
    elif prob_percent >= 60:
        rekomendasi = ("Perlu sedikit peningkatan. Tingkatkan presensi, atur jadwal belajar, "
                       "dan konsultasi dengan dosen PA bila perlu.")
    else:
        rekomendasi = ("Wajib ikut mentoring / bimbingan intensif. Fokus pada peningkatan Indeks Prestasi Semester (IPS), "
                       "kurangi mengulang mata kuliah, dan tingkatkan kehadiran.")

    return label, prob_percent, rekomendasi

# ---------- route utama ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    # Jika model gagal load, sampaikan pesan sederhana (tampilkan di UI)
    if model_load_error:
        error = model_load_error

    if request.method == "POST":
        try:
            # Ambil dan sanitasi input
            nama = request.form.get("nama", "").strip()
            nim = request.form.get("nim", "").strip()

            ipk = to_float(request.form.get("ipk", ""), "ipk")
            sks_lulus = to_int(request.form.get("sks_lulus", "0"), "sks_lulus")
            presensi = to_int(request.form.get("presensi", "0"), "presensi")
            mengulang = to_int(request.form.get("mengulang", "0"), "mengulang")

            # Validasi lebih lanjut
            if not nama:
                raise ValueError("Nama tidak boleh kosong.")
            if not nim:
                raise ValueError("NIM tidak boleh kosong.")
            if not (0.0 <= ipk <= 4.0):
                raise ValueError("IPK harus antara 0.0 - 4.0.")
            if not (0 <= presensi <= 100):
                raise ValueError("Presensi harus antara 0 - 100.")
            if sks_lulus < 0:
                raise ValueError("SKS Lulus harus >= 0.")
            if mengulang < 0:
                raise ValueError("Jumlah mengulang harus >= 0.")

            # Panggil prediksi
            label, prob_percent, rekomendasi = prediksi_kelulusan(
                ipk, sks_lulus, presensi, mengulang
            )

            # Siapkan hasil untuk template
            result = {
                "nama": nama,
                "nim": nim,
                "label": label,
                "probabilitas": f"{prob_percent:.2f}%",
                "prob_value": max(0, min(prob_percent, 100)),  # 0..100 untuk width
                "rekomendasi": rekomendasi,
                "ipk": ipk,
                "sks_lulus": sks_lulus,
                "presensi": presensi,
                "mengulang": mengulang,
            }

            logger.info("Prediksi berhasil untuk NIM=%s: label=%s prob=%.2f", nim, label, prob_percent)

        except Exception as e:
            error = str(e)
            result = None
            logger.warning("Error saat memproses request: %s", error)

    return render_template("index.html", result=result, error=error)

# ---------- jalankan app ----------
if __name__ == "__main__":
    # Jangan nyalakan debug=True di production
    debug_flag = os.getenv("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
    app.run(debug=debug_flag)
