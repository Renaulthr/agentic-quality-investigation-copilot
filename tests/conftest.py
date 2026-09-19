import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """
    Lightweight API test client.

    Application lifespan is intentionally not executed for isolated
    endpoint tests.
    """
    return TestClient(app)