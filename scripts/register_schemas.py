"""
Register Avro schemas with the Redpanda Schema Registry.

Usage:
    python scripts/register_schemas.py
    python scripts/register_schemas.py --url http://localhost:8081

Each schema in schemas/ is registered under the Confluent naming convention:
    <topic>-value   e.g. news-raw-value, price-ticks-value

Run this once after `docker compose up redpanda` before starting the producers.
"""

import argparse
import json
import os
import sys

import requests

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "schemas")

SUBJECTS = [
    ("news-raw-value",     "news_event_v1.avsc"),
    ("price-ticks-value",  "price_tick_v1.avsc"),
]


def load_schema(filename: str) -> str:
    path = os.path.join(SCHEMA_DIR, filename)
    with open(path) as f:
        return f.read().strip()


def register(url: str, subject: str, schema_json: str) -> int:
    endpoint = f"{url}/subjects/{subject}/versions"
    payload = {"schemaType": "AVRO", "schema": schema_json}
    resp = requests.post(endpoint, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()["id"]


def check_existing(url: str, subject: str) -> int | None:
    endpoint = f"{url}/subjects/{subject}/versions/latest"
    resp = requests.get(endpoint, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081"),
                        help="Schema Registry base URL (default: http://localhost:8081)")
    parser.add_argument("--force", action="store_true", help="Re-register even if subject already exists")
    args = parser.parse_args()

    print(f"Schema Registry: {args.url}\n")
    ok = True

    for subject, filename in SUBJECTS:
        schema_json = load_schema(filename)
        existing_id = check_existing(args.url, subject)

        if existing_id is not None and not args.force:
            print(f"  [SKIP]  {subject}  (already registered, id={existing_id})")
            continue

        try:
            schema_id = register(args.url, subject, schema_json)
            print(f"  [OK]    {subject}  -> id={schema_id}")
        except requests.HTTPError as exc:
            print(f"  [FAIL]  {subject}  -> {exc.response.status_code} {exc.response.text}")
            ok = False
        except requests.ConnectionError:
            print(f"  [FAIL]  {subject}  -> cannot connect to {args.url}")
            print("          Is Redpanda running? Try: docker compose up -d redpanda")
            ok = False

    print()
    if ok:
        print("All schemas registered. Producers can now use Avro serialization.")
    else:
        print("Some schemas failed — see errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
