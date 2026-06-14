"""
Clear the LMCache engine via its HTTP management API.

Usage:
    source /home/superadmin/Dev/LMCache/.venv/bin/activate
    python3 cache_utils.py [--url http://localhost:8080]

The LMCacheEngineBuilder.destroy() approach does NOT work from a separate
process — _instances is in-process memory inside the vLLM server. The MP
HTTP server at POST /clear-cache is the correct cross-process interface.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8080"


def destroy_cache(url: str = DEFAULT_URL) -> None:
    target = f"{url.rstrip('/')}/clear-cache"
    req = urllib.request.Request(target, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            print(f"[cache] LMCache cleared: {body}")
    except urllib.error.URLError as e:
        print(f"[cache] Cannot reach {target} — is the vLLM+LMCache server running? ({e.reason})", file=sys.stderr)
        sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"[cache] Server error {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear LMCache via HTTP API")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"LMCache MP HTTP server URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()
    destroy_cache(args.url)
