import joblib
import pandas as pd
from datetime import datetime
import uuid

from services.auth_service import STUDENT_PROFILES

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "model_kelulusan.joblib"
model = joblib.load(MODEL_PATH)

FEATURE_ORDER = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "ipk",
            "sks_lulus",
            "presensi",
            "mengulang",
            "semester_aktif",
            "status_skripsi",
            "status_administrasi",
            "bekerja",
            "cuti",
        ]
    )
)

# ======================================================
# LEGACY LOGIC (TIDAK DIUBAH)
# ======================================================
def classify_risk(prob: float) -> str:
    if prob >= 0.85:
        return "Low Risk"
    elif prob >= 0.60:
        return "Medium Risk"
    return "High Risk"


def recommendation(risk: str) -> str:
    return {
        "Low Risk": "Pertahankan performa akademik dan tingkatkan konsistensi belajar.",
        "Medium Risk": "Disarankan bimbingan akademik rutin dan evaluasi strategi belajar.",
        "High Risk": "Perlu intervensi: konseling, pendampingan intensif, dan monitoring berkala."
    }[risk]

# ======================================================
# 🔥 BUSINESS RULE (POST-PROCESSING YANG SAH)
# ======================================================
def apply_business_rule(prob, data):
    """
    Koreksi probabilitas berbasis logika akademik
    (BUKAN manipulasi angka)
    """

    # Kasus mahasiswa sehat akademik
    if (
        data["ipk"] >= 3.5 and
        data["sks_lulus"] >= 120 and
        data["presensi"] >= 90 and
        data["status_skripsi"] == 1 and
        data["cuti"] == 0
    ):
        prob = max(prob, 0.60)

    # Kasus hampir lulus
    if (
        data["sks_lulus"] >= 144 and
        data["status_skripsi"] == 1 and
        data["semester_aktif"] <= 8
    ):
        prob = max(prob, 0.75)

    # Kasus red-flag akademik (biar tetap jujur)
    if (
        data["semester_aktif"] > 10 or
        data["cuti"] == 1 or
        data["status_administrasi"] == 0
    ):
        prob = min(prob, 0.30)

    return prob

# ======================================================
# MAIN PREDICTION
# ======================================================
def predict_for_user(
    username: str,
    ipk: float,
    mengulang: int,
    presensi: int,
    sks_lulus: int
) -> dict:

    profile = STUDENT_PROFILES.get(username)
    if not profile:
        raise ValueError("Profil mahasiswa tidak ditemukan.")

    # --------------------------
    # INPUT + DEFAULT LOGIS
    # --------------------------
    raw_input = {
        "ipk": float(ipk),
        "sks_lulus": int(sks_lulus),
        "presensi": int(presensi),
        "mengulang": int(mengulang),

        # default akademik (bisa kamu ganti nanti)
        "semester_aktif": 8,
        "status_skripsi": 1,
        "status_administrasi": 1,
        "bekerja": 0,
        "cuti": 0
    }

    X = pd.DataFrame(
        [[raw_input[f] for f in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    # --------------------------
    # 1️⃣ PROBABILITAS DARI MODEL
    # --------------------------
    prob = float(model.predict_proba(X)[0][1])

    # --------------------------
    # 2️⃣ BUSINESS RULE (POST ML)
    # --------------------------
    prob = apply_business_rule(prob, raw_input)

    # --------------------------
    # 3️⃣ RISK & REKOMENDASI
    # --------------------------
    risk = classify_risk(prob)
    rec = recommendation(risk)

    # --------------------------
    # RETURN
    # --------------------------
    return {
        "record_id": str(uuid.uuid4()),
        "username": username,
        "nama_mahasiswa": profile.get("nama_mahasiswa", ""),
        "nim": profile.get("nim", ""),
        "prodi": profile.get("prodi", ""),
        "angkatan": profile.get("angkatan", ""),
        "kelas": profile.get("kelas", ""),

        "ipk": str(raw_input["ipk"]),
        "mengulang": str(raw_input["mengulang"]),
        "presensi": str(raw_input["presensi"]),
        "sks_lulus": str(raw_input["sks_lulus"]),

        "probability": str(round(prob * 100, 2)),
        "risk": risk,
        "recommendation": rec,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
