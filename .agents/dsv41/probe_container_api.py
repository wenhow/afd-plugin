"""Exercise native vLLM routes in the accepted container without a listener."""

import importlib.metadata as metadata
import json
from argparse import Namespace

from fastapi.testclient import TestClient
from vllm.entrypoints.launchers.app import build_app
from vllm.version import __version__ as vllm_version


def main():
    args = Namespace(
        disable_fastapi_docs=False,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=["local-only-smoke"],
        enable_request_id_headers=True,
        middleware=[],
        enable_fault_tolerance=False,
        log_error_stack=False,
    )
    app = build_app(args, supported_tasks=("generate",), model_config=None)
    # Do not enter TestClient: the lifespan requires initialized engine state.
    client = TestClient(app, headers={"Authorization": "Bearer local-only-smoke"})
    try:
        results = {}
        for path in ("/version", "/openapi.json", "/__afd_missing_route__", "/metrics"):
            response = client.get(path)
            expected = 404 if path == "/__afd_missing_route__" else 200
            assert response.status_code == expected, (
                path,
                response.status_code,
                response.text[:1000],
            )
            results[path] = {
                "status": response.status_code,
                "content_type": response.headers.get("content-type"),
            }
            if path == "/version":
                assert response.json()["version"] == vllm_version
                results[path]["version"] = response.json()["version"]
            elif path == "/openapi.json":
                assert "/version" in response.json()["paths"]
            elif path == "/metrics":
                assert "# HELP" in response.text
        print(
            "API_ROUTE_SMOKE "
            + json.dumps(
                {
                    "fastapi": metadata.version("fastapi"),
                    "starlette": metadata.version("starlette"),
                    "instrumentator": metadata.version(
                        "prometheus-fastapi-instrumentator"
                    ),
                    "engine_constructed": False,
                    "model_constructed": False,
                    "network_listener_started": False,
                    "server_lifespan_executed": False,
                    "responses": results,
                }
            ),
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
