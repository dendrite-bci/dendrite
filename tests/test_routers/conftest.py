"""Shared fixtures for router tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from dendrite.web.app import create_app
from dendrite.web.deps import cleanup_services, init_services
from dendrite.web.ws.bridge import QueueBridge


@pytest.fixture
def app():
    """Create app with manually initialized services and bridge."""
    application = create_app()
    init_services()
    bridge = QueueBridge()
    application.state.queue_bridge = bridge
    yield application
    cleanup_services()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
