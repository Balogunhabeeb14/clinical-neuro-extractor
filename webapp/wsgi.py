"""WSGI entrypoint: `gunicorn webapp.wsgi:app` or `python -m webapp.wsgi` for local dev."""

import os

from webapp.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
