"""End-to-end integration tests for core TensorLens API workflows."""


def test_health_check_endpoint(client):
    """Verify health endpoint returns healthy status and metadata."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["product"] == "tensor-lens-ryan-vo"


def test_list_seeded_models_and_hardware(client, analyst_headers):
    """Verify default seeded hardware profiles and models are accessible."""
    hw_resp = client.get("/api/hardware", headers=analyst_headers)
    assert hw_resp.status_code == 200
    hw_data = hw_resp.json()
    assert len(hw_data) >= 4
    assert any("1650" in h["name"] for h in hw_data)

    models_resp = client.get("/api/models", headers=analyst_headers)
    assert models_resp.status_code == 200
    models_data = models_resp.json()
    assert len(models_data) >= 3
    assert any("70B" in m["name"] for m in models_data)


def test_calculate_plan_and_retrieve_history(client, analyst_headers):
    """Verify calculating an inference plan and retrieving it from persistent storage."""
    payload = {
        "model_id": "model-llama3-70b",
        "hardware_id": "gpu-gtx-1650-4gb",
        "quantization": "int4",
        "batch_size": 1,
        "seq_len": 512,
    }
    calc_resp = client.post("/api/planner/calculate", json=payload, headers=analyst_headers)
    assert calc_resp.status_code == 200
    plan = calc_resp.json()
    assert plan["layers_per_chunk"] >= 1
    assert plan["total_chunks"] > 0
    assert plan["vram_utilization_percent"] <= 100.0

    # Retrieve from list endpoint
    list_resp = client.get("/api/planner/plans", headers=analyst_headers)
    assert list_resp.status_code == 200
    plans_list = list_resp.json()
    assert any(p["id"] == plan["id"] for p in plans_list)


def test_simulate_stream_and_record_benchmark(client, analyst_headers):
    """Verify live stream simulation and persistent benchmark execution recording."""
    bench_payload = {
        "model_id": "model-llama3-70b",
        "hardware_id": "gpu-gtx-1650-4gb",
        "quantization": "int4",
        "layers_per_chunk": 1,
        "prompt_tokens": 16,
        "output_tokens": 8,
        "run_numerical_simulation": True,
    }

    # Simulate
    sim_resp = client.post("/api/benchmark/simulate", json=bench_payload, headers=analyst_headers)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert sim_data["chunks_count"] > 0
    assert len(sim_data["steps"]) > 0
    assert sim_data["numerical_check_passed"] is True

    # Record run
    run_resp = client.post("/api/benchmark/run", json=bench_payload, headers=analyst_headers)
    assert run_resp.status_code == 201
    run_data = run_resp.json()
    assert run_data["tokens_per_sec"] > 0
    assert run_data["status"] == "completed"

    # Verify history
    hist_resp = client.get("/api/benchmark/history", headers=analyst_headers)
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()) >= 1


def test_export_benchmarks_csv_and_json(client, analyst_headers):
    """Verify exporting benchmark run telemetry in CSV and JSON formats."""
    # Ensure at least one run exists
    bench_payload = {
        "model_id": "model-mistral-7b",
        "hardware_id": "gpu-t4-16gb",
        "quantization": "int8",
        "layers_per_chunk": 4,
        "prompt_tokens": 16,
        "output_tokens": 8,
    }
    client.post("/api/benchmark/run", json=bench_payload, headers=analyst_headers)

    # Test CSV export
    csv_resp = client.get("/api/benchmark/export?format=csv", headers=analyst_headers)
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert "model_id,hardware_id,quantization" in csv_resp.text

    # Test JSON export
    json_resp = client.get("/api/benchmark/export?format=json", headers=analyst_headers)
    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"].startswith("application/json")
    assert isinstance(json_resp.json(), list)


def test_compatibility_audit_endpoint(client, analyst_headers):
    """Verify compatibility audit endpoint analyzes environment."""
    payload = {
        "model_name": "Llama-2-70B",
        "transformers_version": "4.45.0",
        "torch_version": "2.4.0",
        "target_device": "cuda",
        "head_dim": 128,
        "rotary_dim": 128,
    }
    resp = client.post("/api/compatibility/audit", json=payload, headers=analyst_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_model"] == "Llama-2-70B"
    assert len(data["issues"]) > 0
    assert len(data["recommendations"]) > 0
