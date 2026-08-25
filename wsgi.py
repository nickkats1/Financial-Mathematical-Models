"""WSGI entry point for production servers (e.g. gunicorn)."""

from app import create_app

application = create_app()


if __name__ == "__main__":
    application.run(
        host="0.0.0.0",
        port=application.config["APP_CONFIG"].port,
        debug=False,
    )
