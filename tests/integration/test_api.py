"""HTTP API tests with httpx.AsyncClient against create_app(fake settings).

Cases: POST /v1/ask returns 200 + trace_id header; /v1/ask/stream event order;
GET /v1/explain/<unknown> -> 404; reindex with a bad HMAC -> 401;
/readyz is 503 before the index loads and 200 after.
"""
