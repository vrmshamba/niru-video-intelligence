"""
training/setup_label_studio.py

Creates a Label Studio project for NIRU with:
- 20 Nairobi class bounding-box labels
- All 50 prioritised frames imported as tasks

Usage (run once after starting Label Studio):
    python training/setup_label_studio.py
"""

import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

LS_URL  = "http://localhost:8080"
EMAIL   = "admin@niru.local"
PASSWORD = "niru1234"

THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
FRAMES_DIR  = PROJECT_DIR / "annotation_frames"
FRAMES_LIST = THIS_DIR / "frames_to_label.txt"

NAIROBI_CLASSES = [
    "matatu", "nganya", "city_hoppa", "school_bus", "large_bus",
    "bodaboda", "bodaboda_no_helmet", "bodaboda_stage", "tuk_tuk",
    "mkokoteni", "street_hawkers", "traffic_marshal", "conductor",
    "matatu_stage", "passenger", "illegal_dumping", "garbage_pile",
    "traffic_lights", "road_sign", "pothole",
]

# Label colours — cycling through a distinct palette
COLOURS = [
    "#FF6B35","#F7C59F","#EFEFD0","#004E89","#1A936F",
    "#88D498","#C6DABF","#FFC857","#DB3A34","#323031",
    "#3F88C5","#44BBA4","#E94F37","#393E41","#F5A623",
    "#7B2D8B","#2ECC71","#E74C3C","#3498DB","#F39C12",
]


def api(method: str, path: str, data=None, token: str = "") -> dict:
    url = LS_URL + path
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body[:300]}")


def get_token() -> str:
    """Sign up (first run) or sign in, return API token."""
    # Try sign-up first
    try:
        result = api("POST", "/api/user/signup", {
            "email": EMAIL, "password": PASSWORD,
            "first_name": "NIRU", "last_name": "Admin",
        })
        token = result.get("token", "")
        if token:
            print(f"[setup] Account created: {EMAIL}")
            return token
    except RuntimeError:
        pass  # already exists

    # Sign in
    result = api("POST", "/user/login", {"email": EMAIL, "password": PASSWORD})
    # Login returns redirect; get token via API
    result = api("POST", "/api/user/token", {"email": EMAIL, "password": PASSWORD})
    token = result.get("token", "")
    if not token:
        # Try the auth endpoint
        result = api("GET", f"/api/current-user/token")
    return token


def get_token_v2() -> str:
    """Get token using the correct Label Studio endpoint."""
    # Create account (ignore error if exists)
    try:
        api("POST", "/api/user/signup", {
            "email": EMAIL, "password": PASSWORD,
            "first_name": "NIRU", "last_name": "Admin",
        })
        print(f"[setup] Account created: {EMAIL}")
    except RuntimeError as e:
        if "already" in str(e).lower() or "400" in str(e) or "exists" in str(e).lower():
            print(f"[setup] Account already exists, signing in...")
        else:
            print(f"[setup] Signup note: {e}")

    # Get token
    result = api("GET", "/api/current-user/token", token="")
    # That won't work without auth — use session-based login
    # Use the admin API token endpoint via password auth
    result = api("POST", "/api/auth/token", {"email": EMAIL, "password": PASSWORD})
    return result.get("token", "")


def build_label_config() -> str:
    """Build Label Studio XML config for bounding-box detection."""
    choices = "\n".join(
        f'    <Label value="{cls}" background="{COLOURS[i % len(COLOURS)]}"/>'
        for i, cls in enumerate(NAIROBI_CLASSES)
    )
    return f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image" showInline="true">
{choices}
  </RectangleLabels>
</View>"""


def read_image_paths() -> list[str]:
    """Read absolute image paths from frames_to_label.txt."""
    paths = []
    in_paths = False
    for line in FRAMES_LIST.read_text().splitlines():
        if line.strip() == "Image paths:":
            in_paths = True
            continue
        if in_paths and line.strip().endswith(".jpg"):
            paths.append(line.strip())
    return paths


def main():
    print(f"[setup] Connecting to Label Studio at {LS_URL} ...")

    # --- authenticate ---
    # Try direct token endpoint after signup
    try:
        api("POST", "/api/user/signup", {
            "email": EMAIL, "password": PASSWORD,
            "first_name": "NIRU", "last_name": "Admin",
        })
        print(f"[setup] Account created: {EMAIL}  password: {PASSWORD}")
    except RuntimeError as e:
        print(f"[setup] (account exists or signup skipped: {str(e)[:80]})")

    # Get token via login
    try:
        resp = api("POST", "/api/current-user/reset-token", {
            "email": EMAIL, "password": PASSWORD
        })
        token = resp.get("token", "")
    except RuntimeError:
        token = ""

    if not token:
        try:
            resp = api("POST", "/user/login", {"email": EMAIL, "password": PASSWORD})
            token = resp.get("token", "")
        except RuntimeError:
            token = ""

    if not token:
        print("[setup] Could not auto-retrieve token.")
        print(f"        Open {LS_URL} in your browser, log in with:")
        print(f"          email:    {EMAIL}")
        print(f"          password: {PASSWORD}")
        print("        Then go to Account & Settings > Access Token and paste it below.")
        token = input("Paste your API token: ").strip()

    print(f"[setup] Token acquired: {token[:8]}...")

    # --- create project ---
    label_config = build_label_config()
    project = api("POST", "/api/projects", {
        "title": "NIRU — Nairobi Street Annotation",
        "description": "Manual annotation of 20 custom Nairobi classes for YOLOv8 fine-tuning.",
        "label_config": label_config,
    }, token=token)
    project_id = project["id"]
    print(f"[setup] Project created: id={project_id}  '{project['title']}'")

    # --- import tasks ---
    image_paths = read_image_paths()
    if not image_paths:
        print("[setup] ERROR: No image paths found in frames_to_label.txt")
        return

    # Label Studio local file serving: use /data/local-files/?d=<path>
    # We need to serve images — use absolute file URIs
    tasks = []
    for p in image_paths:
        # Convert Windows path to forward-slash URI
        uri = "file:///" + p.replace("\\", "/").replace(" ", "%20")
        tasks.append({"data": {"image": uri}})

    result = api("POST", f"/api/projects/{project_id}/import", tasks, token=token)
    imported = result.get("task_count", len(tasks))
    print(f"[setup] Imported {imported} tasks into project {project_id}.")

    print(f"\n[setup] Done! Open your browser:")
    print(f"  {LS_URL}/projects/{project_id}/")
    print(f"\n  email:    {EMAIL}")
    print(f"  password: {PASSWORD}")
    print(f"\nNote: If images show as broken, enable local file serving:")
    print(f"  Set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
    print(f"  and LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=H:/")
    print(f"  before starting Label Studio.")


if __name__ == "__main__":
    main()
