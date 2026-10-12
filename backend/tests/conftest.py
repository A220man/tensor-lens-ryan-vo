"""Pytest fixtures and configuration for TensorLens tests."""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure test environment
os.environ["ENVIRONMENT"] = "development"
os.environ["DEMO_MODE"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.core.database import Base, get_db
from app.services.storage import seed_defaults
from app.core.auth import create_demo_token
from app.main import app

# Test SQLite in-memory database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    future=True,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine, future=True)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables in memory once for the test session."""
    Base.metadata.create_all(bind=test_engine)
    with TestingSessionLocal() as session:
        seed_defaults(session)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Provide a fresh transactional session for test isolation."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    """Provide a TestClient with database session override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def analyst_headers():
    """Authorization headers for Analyst role."""
    token = create_demo_token(role="analyst")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers():
    """Authorization headers for Viewer role."""
    token = create_demo_token(role="viewer")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers():
    """Authorization headers for Admin role."""
    token = create_demo_token(role="admin")
    return {"Authorization": f"Bearer {token}"}
