"""Flask inference service.  gunicorn 'regml.api:create_app()'"""
from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, jsonify, request

from regml.config import load_config, setup_logging
from regml.predict import Predictor, ValidationError

log = logging.getLogger("regml.api")


def create_app(predictor: Predictor | None = None, cfg: dict | None = None) -> Flask:
    setup_logging()
    cfg = cfg or load_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    max_batch = cfg["api"]["max_batch_size"]
    state = {"predictor": predictor}

    def get_predictor() -> Predictor:
        if state["predictor"] is None:
            state["predictor"] = Predictor(cfg["artifacts"]["model_dir"])
        return state["predictor"]

    @app.before_request
    def _start():
        g.t0, g.rid = time.perf_counter(), request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

    @app.after_request
    def _log(resp):
        ms = (time.perf_counter() - g.t0) * 1000
        resp.headers["X-Request-ID"] = g.rid
        log.info("rid=%s %s %s -> %s (%.1f ms)", g.rid, request.method, request.path, resp.status_code, ms)
        return resp

    @app.get("/health")
    def health():  # liveness: process is up
        return jsonify(status="ok")

    @app.get("/ready")
    def ready():  # readiness: model can be loaded
        try:
            get_predictor()
            return jsonify(status="ready")
        except Exception:
            log.exception("model not ready")
            return jsonify(status="not_ready"), 503

    @app.get("/metadata")
    def metadata():
        m = get_predictor().metadata
        keys = ["model_version", "model_name", "trained_at", "input_columns",
                "required_fields", "test_metrics", "sklearn"]
        return jsonify({k: m.get(k) for k in keys})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Body must be valid JSON.")
        records = payload["instances"] if isinstance(payload, dict) and "instances" in payload else payload
        if isinstance(records, list) and len(records) > max_batch:
            raise ValidationError(f"Batch too large (max {max_batch}).")
        p = get_predictor()
        preds = p.predict(records)
        return jsonify(predictions=preds, count=len(preds), model_version=p.metadata["model_version"])

    @app.errorhandler(ValidationError)
    def _bad(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(404)
    def _nf(e):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def _na(e):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(413)
    def _big(e):
        return jsonify(error="Payload too large"), 413

    @app.errorhandler(Exception)
    def _err(e):
        log.exception("unhandled error")
        return jsonify(error="Internal server error"), 500

    return app
