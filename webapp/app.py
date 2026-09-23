"""Flask app: neuropsychology patient registry viewer."""

from __future__ import annotations

import os

import pandas as pd
from flask import Flask, render_template, request

from clinical_neuro_extractor.registry import build_patient_registry

from . import sources


def _prepare_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Stringify list/datetime cells so the table renders cleanly."""
    display_df = df.copy()
    for col in display_df.columns:
        if display_df[col].apply(lambda v: isinstance(v, list)).any():
            display_df[col] = display_df[col].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
            )
    datetime_cols = display_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        display_df[col] = display_df[col].dt.strftime("%Y-%m-%d").fillna("")
    return display_df.fillna("")


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

    @app.route("/", methods=["GET", "POST"])
    def index():
        uploaded_file = None
        force_refresh = False
        deidentify = False

        if request.method == "POST":
            upload = request.files.get("csv_file")
            if upload and upload.filename:
                uploaded_file = upload
            force_refresh = request.form.get("refresh") == "1"
            deidentify = request.form.get("deidentify") == "1"

        try:
            documents, source_label, warning = sources.resolve_documents(
                uploaded_file=uploaded_file,
                force_refresh=force_refresh,
            )
        except sources.CogStackUnavailable as exc:
            return render_template("index.html", error=str(exc))

        if documents.empty:
            return render_template(
                "index.html", error=None, empty=True, source_label=source_label
            )

        salt = os.environ.get("REGISTRY_SALT", "")
        registry = build_patient_registry(documents, deidentify=deidentify, salt=salt)
        display_df = _prepare_for_display(registry)

        return render_template(
            "index.html",
            error=None,
            warning=warning,
            columns=list(display_df.columns),
            rows=display_df.to_dict(orient="records"),
            source_label=source_label,
            deidentify=deidentify,
            n_documents=len(documents),
            n_patients=len(registry),
        )

    return app
