def test_api_import_and_app():
    from src.api.app import app
    assert app.title == "Supply Chain AI Monitoring API"
