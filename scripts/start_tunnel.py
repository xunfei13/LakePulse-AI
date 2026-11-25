"""Launch Cloudflare Quick Tunnel and auto-update Dify openapi_tools.json."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from lakepulse.api.agent_workflow import export_dify_openapi_schema
from lakepulse.config import AppSettings, get_settings

TRYCLOUDFLARE_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _locate_cloudflared(cfg: AppSettings) -> Path:
    """Locate bundled or system cloudflared executable."""
    exe_name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    bundled = cfg.project_root / "artifacts" / "tools" / exe_name
    if bundled.is_file():
        return bundled
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"cloudflared executable not found at {bundled} or on PATH."
    )


def _check_local_gateway(local_url: str) -> None:
    """Ensure local FastAPI server is responding on /health before opening tunnel."""
    health_url = f"{local_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3.0) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Unexpected status {resp.status} from {health_url}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(
            f"Local FastAPI gateway is not reachable at {health_url}. "
            "Start it first with '.\\make.ps1 serve'."
        ) from exc


def main() -> int:
    """Run cloudflared tunnel, capture public URL, update OpenAPI JSON, and stream logs."""
    parser = argparse.ArgumentParser(
        description="Start Cloudflare Quick Tunnel and sync docs/dify/openapi_tools.json."
    )
    parser.add_argument(
        "--local-url",
        default=None,
        help="Local FastAPI URL (defaults to http://API_HOST:API_PORT).",
    )
    args = parser.parse_args()

    cfg = get_settings()
    local_url = args.local_url or f"http://{cfg.api_host}:{cfg.api_port}"
    _check_local_gateway(local_url)
    cloudflared_bin = _locate_cloudflared(cfg)

    cmd = [
        str(cloudflared_bin),
        "tunnel",
        "--url",
        local_url,
        "--protocol",
        "http2",
        "--no-autoupdate",
    ]
    print(f"[Tunnel] Starting Cloudflare Quick Tunnel -> {local_url}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    updated = False
    openapi_path = cfg.project_root / "docs" / "dify" / "openapi_tools.json"
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not updated:
                match = TRYCLOUDFLARE_RE.search(line)
                if match:
                    public_url = match.group(0)
                    export_dify_openapi_schema(
                        openapi_path, server_url=public_url, settings=cfg
                    )
                    updated = True
                    print(
                        f"\n[Tunnel] Public URL active: {public_url}\n"
                        f"[Tunnel] Updated {openapi_path} -> servers[0].url={public_url}\n"
                        "[Tunnel] Keep this window open while using Dify Cloud (Ctrl+C to stop).\n"
                    )
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[Tunnel] Stopping Cloudflare Tunnel...")
        proc.terminate()
        proc.wait(timeout=5)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
