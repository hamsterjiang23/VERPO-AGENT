"""Session service isolates ALFWorld/WebShop runtimes from veRL/CUDA workers."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

from .backends import make_factory
from .data import read_manifest, require_provenance
from .protocol import BENCHMARKS
from verpo_agent.provenance import atomic_json, digest, ROOT


def load_service_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ("manifest", "output_dir", "resource_manifest"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Environment service requires {key}")
        config[key] = str((path.parent / config[key]).resolve())
    if config.get("benchmark") not in BENCHMARKS:
        raise ValueError("Unknown benchmark")
    for key in ("max_sessions", "session_ttl_seconds"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Service {key} must be positive")
    backend = config["backend"]
    for key in (
        "source_root",
        "data_root",
        "alfworld_config",
        "products_file",
        "attributes_file",
    ):
        if key in backend:
            if not isinstance(backend[key], str) or not backend[key]:
                raise ValueError(f"backend.{key} must be configured")
            backend[key] = str((path.parent / backend[key]).resolve())
    return config


def verify_service(config):
    rows = read_manifest(config["manifest"])
    if any(
        row["benchmark"] != config["benchmark"]
        for split in rows.values()
        for row in split
    ):
        raise ValueError("Service benchmark disagrees with episodes")
    resources_path = Path(config["resource_manifest"])
    resources = json.loads(resources_path.read_text())
    require_provenance(resources.get("provenance"))
    if not resources.get("provenance") or not resources.get("files"):
        raise ValueError("Register actual benchmark resources and provenance")
    for name, sha in resources["files"].items():
        if digest(resources_path.parent / name) != sha:
            raise ValueError(f"Benchmark resource hash mismatch: {name}")
    registered = {
        (resources_path.parent / name).resolve() for name in resources["files"]
    }
    backend = config["backend"]
    if config["benchmark"] == "search_qa":
        from urllib.parse import urlparse

        require_provenance(backend.get("retriever_provenance"))
        required = {"model_revision", "corpus_revision", "index_sha256"}
        if not required <= backend["retriever_provenance"].keys():
            raise ValueError("Register retriever model, corpus and index provenance")
        if (
            urlparse(backend["search_url"]).scheme not in {"http", "https"}
            or backend["topk"] != 3
            or backend["timeout_seconds"] <= 0
        ):
            raise ValueError(
                "Search-QA requires an HTTP Search-R1 service, topk=3 and a timeout"
            )
    elif config["benchmark"] == "alfworld":
        if Path(backend["alfworld_config"]).resolve() not in registered:
            raise ValueError("Register the actual ALFWorld config resource")
        root = Path(backend["data_root"]).resolve()
        for split in rows.values():
            for row in split:
                game = (root / row["gamefile"]).resolve()
                if not game.is_relative_to(root) or digest(game) != row["game_sha256"]:
                    raise ValueError("ALFWorld game resource checksum mismatch")
    else:
        required_files = {
            Path(backend[k]).resolve() for k in ("products_file", "attributes_file")
        }
        index = (
            Path(backend["source_root"])
            / "agent_system/environments/env_package/webshop/webshop/search_engine/indexes"
        )
        index_files = {p.resolve() for p in index.rglob("*") if p.is_file()}
        if not index_files or not (required_files | index_files) <= registered:
            raise ValueError(
                "Register WebShop products, attributes and every actual Lucene index file"
            )
    packages = config.get("package_versions", {})
    if not packages or any(
        importlib.metadata.version(k) != v for k, v in packages.items()
    ):
        raise ValueError("Register exact installed environment package versions")
    if config["benchmark"] != "search_qa":
        from .sources import verify_source

        verify_source(config["backend"]["source_root"])
    files = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src/verpo_agent/benchmarks").glob("*.py"))
    }
    identity = {
        "config": config,
        "manifest_sha256": digest(config["manifest"]),
        "resource_sha256": digest(resources_path),
        "source": files,
    }
    return rows, hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()


class SessionService:
    def __init__(self, config, rows, identity, factory):
        self.config, self.identity, self.factory = config, identity, factory
        self.rows = {r["task_id"]: r for split in rows.values() for r in split}
        self.sessions, self.lock = {}, threading.RLock()

    def health(self):
        return {
            "benchmark": self.config["benchmark"],
            "identity": self.identity,
            "manifest_sha256": digest(self.config["manifest"]),
            "protocol_version": 1,
        }

    def create(self, task_id, seed):
        with self.lock:
            self.expire()
            if len(self.sessions) >= self.config["max_sessions"]:
                raise ValueError("Environment session capacity exhausted")
            row = self.rows[task_id]
            env = self.factory(row, seed)
            try:
                observation, info = env.reset()
            except BaseException:
                env.close()
                raise
            key = uuid4().hex
            self.sessions[key] = {
                "env": env,
                "row": row,
                "turn": 0,
                "done": False,
                "touched": time.monotonic(),
                "lock": threading.RLock(),
            }
            result = {
                "session_id": key,
                "observation": observation,
                "available_actions": info.get("available_actions"),
                "identity": self.identity,
            }
            atomic_json(
                Path(self.config["output_dir"]) / key / "reset.json",
                {"task_id": task_id, "seed": seed, **result},
            )
            return result

    def step(self, key, action, turn):
        session = self.sessions[key]
        with session["lock"]:
            if session["done"] or turn != session["turn"]:
                raise ValueError("Terminated session or duplicate/out-of-order action")
            try:
                observation, reward, done, info = session["env"].step(action)
            except Exception as error:
                session["done"] = True
                atomic_json(
                    Path(self.config["output_dir"]) / key / f"turn_{turn}.error.json",
                    {
                        "raw_action": action,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
                raise
            session["turn"] += 1
            limited = (
                session["turn"] >= BENCHMARKS[self.config["benchmark"]]
                and not reward
                and (not done or info.get("time_limit", False))
            )
            session["done"] = done or limited
            session["touched"] = time.monotonic()
            result = {
                "observation": observation,
                "reward": reward,
                "done": session["done"],
                "termination": "turn_limit"
                if limited
                else "final"
                if done
                else "running",
                "valid_action": bool(info.get("valid_action", True)),
                "score": float(info.get("task_score", reward)),
                "available_actions": info.get("available_actions"),
                "details": info,
            }
            if reward not in (0.0, 1.0):
                raise ValueError("Expected binary benchmark success reward")
            atomic_json(
                Path(self.config["output_dir"]) / key / f"turn_{turn}.json",
                {"raw_action": action, **result},
            )
            return result

    def release(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
            if session:
                with session["lock"]:
                    session["env"].close()

    def expire(self):
        for key, value in list(self.sessions.items()):
            if time.monotonic() - value["touched"] > self.config["session_ttl_seconds"]:
                self.release(key)


def create_app(service):
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    @app.get("/health")
    def health():
        return service.health()

    @app.post("/sessions")
    def create(body: dict):
        try:
            return service.create(body["task_id"], body["seed"])
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/sessions/{session_id}/step")
    def step(session_id: str, body: dict):
        try:
            return service.step(session_id, body["action"], body["turn"])
        except (ValueError, KeyError) as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/sessions/{session_id}")
    def release(session_id: str):
        service.release(session_id)
        return {"released": True}

    return app


def serve(path, host, port):
    import uvicorn

    config = load_service_config(path)
    rows, identity = verify_service(config)
    service = SessionService(
        config, rows, identity, make_factory(config["benchmark"], config["backend"])
    )
    atomic_json(Path(config["output_dir"]) / "service_identity.json", service.health())
    try:
        uvicorn.run(create_app(service), host=host, port=port)
    finally:
        for key in list(service.sessions):
            service.release(key)
