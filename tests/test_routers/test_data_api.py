"""Tests for data explorer REST endpoints."""



# --- Studies ---


async def test_list_studies_returns_200(client):
    resp = await client.get("/api/data/studies")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_create_study(client):
    resp = await client.post(
        "/api/data/studies",
        json={"study_name": "test_study", "description": "A test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["study_name"] == "test_study"
    assert data["study_id"] is not None


async def test_get_study_detail(client):
    # Create a study first
    create_resp = await client.post(
        "/api/data/studies", json={"study_name": "detail_test"}
    )
    study_id = create_resp.json()["study_id"]

    resp = await client.get(f"/api/data/studies/{study_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["study_name"] == "detail_test"
    assert data["recording_count"] == 0


async def test_get_study_not_found(client):
    resp = await client.get("/api/data/studies/99999")
    assert resp.status_code == 404


async def test_delete_study(client):
    create_resp = await client.post(
        "/api/data/studies", json={"study_name": "delete_me"}
    )
    study_id = create_resp.json()["study_id"]

    resp = await client.delete(f"/api/data/studies/{study_id}")
    assert resp.status_code == 200

    resp = await client.get(f"/api/data/studies/{study_id}")
    assert resp.status_code == 404


# --- Recordings ---


async def test_list_recordings_returns_200(client):
    resp = await client.get("/api/data/recordings")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_get_recording_not_found(client):
    resp = await client.get("/api/data/recordings/99999")
    assert resp.status_code == 404


# --- Decoders ---


async def test_list_decoders_returns_200(client):
    resp = await client.get("/api/data/decoders")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_get_decoder_not_found(client):
    resp = await client.get("/api/data/decoders/99999")
    assert resp.status_code == 404
