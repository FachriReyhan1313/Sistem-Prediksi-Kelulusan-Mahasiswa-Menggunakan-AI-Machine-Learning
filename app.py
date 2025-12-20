import os
from flask import Flask
from controllers.auth_controller import auth_bp
from controllers.mahasiswa_controller import mahasiswa_bp
from controllers.dosen_controller import dosen_bp
from controllers.akademik_controller import akademik_bp
from extensions import db

DEFAULT_DB_URL = "postgresql://postgres:postgres@db.ltrpqqaagkhzqfatyvir.supabase.co:5432/db_prediksi_kelulusan"


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "prediksi-kelulusan-secret")

    # DATABASE
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL",
        DEFAULT_DB_URL
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(mahasiswa_bp)
    app.register_blueprint(dosen_bp)
    app.register_blueprint(akademik_bp)

    return app


# ⬇️ INI YANG DIPAKAI VERCEL
app = create_app()

# ❌ JANGAN ADA app.run() DI PRODUCTION
# if __name__ == "__main__":
#     app.run(debug=True)
