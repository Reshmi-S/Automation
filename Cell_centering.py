from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

# Suppress known non-critical PyTorch warning from some scripted models.
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
)

# Reduce OpenCV backend probe noise in console.
try:
    cv2.setLogLevel(0)
except Exception:
    pass


BACKENDS = []
if sys.platform == "win32":
    BACKENDS = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("AUTO", cv2.CAP_ANY),
    ]
else:
    BACKENDS = [("AUTO", cv2.CAP_ANY)]


@contextlib.contextmanager
def _suppress_cv2_stderr():
    """Temporarily silence native OpenCV backend warnings on stderr/stdout."""
    saved = {}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            saved[fd] = os.dup(fd)
            os.dup2(devnull, fd)
        os.close(devnull)
    except Exception:
        saved = {}
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        for fd, old_fd in saved.items():
            try:
                os.dup2(old_fd, fd)
                os.close(old_fd)
            except Exception:
                pass


def load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_working_cameras(max_index: int = 9):
    found = []
    for idx in range(max_index + 1):
        opened = False
        used_backend = ""
        for backend_name, backend in BACKENDS:
            with _suppress_cv2_stderr():
                cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            time.sleep(0.15)
            with _suppress_cv2_stderr():
                ret, _ = cap.read()
            cap.release()
            if ret:
                opened = True
                used_backend = backend_name
                break
        if opened:
            found.append((idx, used_backend))
    return found


def select_camera(default_index: int | None = None, max_index: int = 9):
    cams = find_working_cameras(max_index=max_index)
    if not cams:
        raise RuntimeError(f"No working cameras found at indices 0-{max_index}.")

    print("Available cameras:")
    for idx, backend in cams:
        print(f"  {idx} ({backend})")

    if default_index is not None:
        for idx, _backend in cams:
            if idx == default_index:
                return default_index

    while True:
        raw = input("Select camera index: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a valid integer camera index.")
            continue
        if any(idx == cidx for cidx, _ in cams):
            return idx
        print("That index is not in the detected camera list.")


def open_camera(camera_index: int):
    for backend_name, backend in BACKENDS:
        with _suppress_cv2_stderr():
            cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        time.sleep(0.15)
        with _suppress_cv2_stderr():
            ret, _ = cap.read()
        if ret:
            print(f"[INFO] Camera {camera_index} opened via {backend_name}")
            return cap
        cap.release()
    raise RuntimeError(f"Failed to open camera index {camera_index}.")


def _validate_webcam_frame(cap) -> bool:
    """Try a few reads and validate frame shape/dtype."""
    for _ in range(4):
        try:
            with _suppress_cv2_stderr():
                ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            if frame.ndim == 3 and frame.dtype == np.uint8:
                return True
        except Exception:
            time.sleep(0.02)
            continue
    return False


def configure_webcam_capture(cap, desired_w: int, desired_h: int):
    """Set a stable webcam mode (MJPG + fallback resolutions)."""
    # MJPG often avoids MSMF/driver decode issues on high resolutions.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    candidates = []
    first = (int(desired_w), int(desired_h))
    if first[0] > 0 and first[1] > 0:
        candidates.append(first)
    for wh in ((1920, 1080), (1280, 720), (640, 480)):
        if wh not in candidates:
            candidates.append(wh)

    for w, h in candidates:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        except Exception:
            pass

        if _validate_webcam_frame(cap):
            try:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception:
                aw, ah = w, h
            print(f"[INFO] Webcam mode set to {aw}x{ah}")
            return cap

    raise RuntimeError("Unable to configure webcam to a stable mode.")


def reopen_webcam(cap, camera_index: int, cam_w: int, cam_h: int):
    """Release and reopen webcam with configured resolution and short warm-up."""
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    new_cap = open_camera(int(camera_index))
    return configure_webcam_capture(new_cap, int(cam_w), int(cam_h))


def choose_capture_source_auto(webcam_available: bool, flir_available: bool) -> str:
    """Interactive source picker used when --source=auto.
    Returns 'webcam' or 'flir'."""
    if not webcam_available and not flir_available:
        raise RuntimeError(
            "No webcam detected and FLIR python path is not configured. "
            "Pass --flir-python or set FLIR_PYTHON_PATH."
        )

    # Non-interactive sessions pick the safest available source.
    if not sys.stdin or not sys.stdin.isatty():
        if webcam_available:
            return "webcam"
        return "flir"

    print("Select capture source:")
    print(f"  1) Webcam ({'available' if webcam_available else 'not available'})")
    print(f"  2) FLIR   ({'available' if flir_available else 'not configured'})")

    default_choice = "1" if webcam_available else "2"
    while True:
        raw = input(f"Enter 1 or 2 [default {default_choice}]: ").strip()
        if not raw:
            raw = default_choice
        if raw == "1":
            if webcam_available:
                return "webcam"
            print("Webcam is not available. Choose FLIR.")
            continue
        if raw == "2":
            if flir_available:
                return "flir"
            print("FLIR is not configured. Pass --flir-python or set FLIR_PYTHON_PATH.")
            continue
        print("Please enter 1 or 2.")


def _get_flir_python_path(root_dir: Path, explicit_path: str | None = None) -> str:
    """Resolve FLIR python path from arg, env var, app config, or fallback paths."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("FLIR_PYTHON_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    # Match Main_live.py behavior: read AppData SolarCellAnalyzer config.
    if sys.platform == "win32":
        app_cfg = Path(os.environ.get("APPDATA", "")) / "SolarCellAnalyzer" / "config.json"
    elif sys.platform == "darwin":
        app_cfg = Path.home() / "Library" / "Application Support" / "SolarCellAnalyzer" / "config.json"
    else:
        app_cfg = Path.home() / ".config" / "SolarCellAnalyzer" / "config.json"
    if app_cfg.exists():
        try:
            with open(app_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            flir_path = str(cfg.get("flir_python_path", "") or "").strip()
            if flir_path:
                candidates.append(flir_path)
        except Exception:
            pass

    for cfg_path in (root_dir / "config.json", root_dir / "config" / "config.json"):
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            flir_path = str(cfg.get("flir_python_path", "") or "").strip()
            if flir_path:
                candidates.append(flir_path)
        except Exception:
            pass

    # Practical local fallbacks for this workspace layout.
    candidates.extend([
        str((root_dir.parent / "Flir_detection" / "VISION" / "Scripts" / "python.exe").resolve()),
        str((root_dir / "SCA_CPU" / "Scripts" / "python.exe").resolve()),
    ])

    for path in candidates:
        if path and Path(path).exists():
            return path
    return ""


def start_flir_bridge(root_dir: Path, flir_python: str | None = None, bridge_script: str = "flir_live_bridge.py"):
    bridge_path = Path(bridge_script)
    if not bridge_path.is_absolute():
        bridge_path = (root_dir / bridge_path).resolve()
    if not bridge_path.exists():
        raise FileNotFoundError(f"FLIR bridge script not found: {bridge_path}")

    flir_py = _get_flir_python_path(root_dir, explicit_path=flir_python)
    if not flir_py:
        raise RuntimeError(
            "FLIR python path is not configured. Pass --flir-python or set FLIR_PYTHON_PATH."
        )

    proc = subprocess.Popen(
        [flir_py, str(bridge_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready = proc.stdout.readline().strip() if proc.stdout else ""
    if not ready.startswith("READY"):
        err = ready or "No READY response from bridge"
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"FLIR bridge failed to start: {err}")
    print("[INFO] FLIR bridge started.")
    return proc


def send_flir_bridge_command(proc, command: str) -> bool:
    """Send one command to FLIR bridge and consume one-line response."""
    if proc is None or proc.stdin is None or proc.stdout is None:
        return False
    try:
        proc.stdin.write(f"{command}\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        if line.startswith("OK"):
            return True
        if line:
            print(f"[FLIR CMD] {command} -> {line}")
        return False
    except Exception as e:
        print(f"[FLIR CMD] {command} failed: {e}")
        return False


def apply_flir_camera_settings(proc, cfg: dict):
    """Apply optional FLIR exposure/gain lock settings from config."""
    flir_cam_cfg = dict(cfg.get("flir_camera_settings", {}) or {})
    if not flir_cam_cfg or not bool(flir_cam_cfg.get("enabled", True)):
        return

    exposure_auto = bool(flir_cam_cfg.get("exposure_auto", False))
    gain_auto = bool(flir_cam_cfg.get("gain_auto", False))
    exposure_us = flir_cam_cfg.get("exposure_us", None)
    gain_db = flir_cam_cfg.get("gain_db", None)

    if exposure_auto:
        send_flir_bridge_command(proc, "EXPOSURE_AUTO")
    elif exposure_us is not None:
        send_flir_bridge_command(proc, f"EXPOSURE {float(exposure_us)}")

    if gain_auto:
        send_flir_bridge_command(proc, "GAIN_AUTO")
    elif gain_db is not None:
        send_flir_bridge_command(proc, f"GAIN {float(gain_db)}")


def read_flir_frame_bgr(proc, cfg) -> np.ndarray | None:
    """Request one frame from bridge and decode it to BGR numpy frame."""
    if proc.stdin is None or proc.stdout is None:
        return None

    last_err = ""
    for _ in range(5):
        proc.stdin.write("FRAME RAW\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()

        if not line:
            time.sleep(0.03)
            continue

        if line.startswith("ERR"):
            last_err = line
            if "No frame available" in line:
                time.sleep(0.03)
                continue
            print(f"[FLIR] {line}")
            return None

        # ✅ RAW path
        if line.startswith("RAW "):
            parts = line.split(" ", 4)
            if len(parts) < 5:
                time.sleep(0.03)
                continue
            try:
                h = int(parts[1])
                w = int(parts[2])
                c = int(parts[3])
                raw_bytes = base64.b64decode(parts[4])
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)

                expected = h * w * c
                if arr.size != expected:
                    print(f"[FLIR] RAW size mismatch: got={arr.size}, expected={expected}")
                    time.sleep(0.03)
                    continue

                frame = arr.reshape((h, w, c))


                if cfg.get("flir_preprocess", {}).get("enabled", False):
                    frame = preprocess_flir_frame(frame, cfg["flir_preprocess"])

                return frame

            except Exception as exc:
                print(f"[FLIR] RAW decode error: {exc}")
                return None

        # ✅ JPEG fallback path
        if not line.startswith("FRAME "):
            print(f"[FLIR] Unexpected bridge response: {line}")
            return None

        parts = line.split(" ", 2)
        if len(parts) < 3:
            time.sleep(0.03)
            continue

        try:
            jpeg_bytes = base64.b64decode(parts[2])
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            # ✅ ✅ ✅ ADD HERE ALSO
            if cfg.get("flir_preprocess", {}).get("enabled", False):
                frame = preprocess_flir_frame(frame, cfg["flir_preprocess"])

            return frame

        except Exception as exc:
            print(f"[FLIR] Frame decode error: {exc}")
            return None

    if last_err and "No frame available" not in last_err:
        print(f"[FLIR] {last_err}")

    return None



def preprocess_flir_frame(frame, pp_cfg):
    if frame is None or frame.size == 0:
        return frame

    # Resize
    if pp_cfg.get("resize_width", 0) > 0:
        frame = cv2.resize(frame, (
            pp_cfg["resize_width"],
            pp_cfg["resize_height"]
        ))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Smooth noise a bit
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # ✅ KEY: equalization
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    # ✅ Try invert (VERY IMPORTANT)
    gray = cv2.bitwise_not(gray)

    # CLAHE
    if pp_cfg.get("clahe", False):
        gray = cv2.medianBlur(gray, 5)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )
        gray = clahe.apply(gray)

    # Convert back to 3 channel
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return out




def stop_flir_bridge(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def resolve_weight_path(cfg: dict, script_dir: Path) -> str:
    raw = str(cfg.get("weight_path", "") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (script_dir / p).resolve()
        # Only allow actual files (not directories) for TorchScript load.
        if p.exists() and p.is_file():
            return str(p)
        if p.exists() and p.is_dir():
            print(f"[WARN] weight_path points to a directory, not a model file: {p}")

    for fallback in ("model.ts", "model_gpu.ts", "model_cpu.ts"):
        p = (script_dir / fallback).resolve()
        if p.exists() and p.is_file():
            print(f"[INFO] Using fallback weights: {p}")
            return str(p)

    # As a final fallback, pick the first .ts file in the script directory.
    ts_candidates = sorted(script_dir.glob("*.ts"))
    for p in ts_candidates:
        if p.is_file():
            print(f"[INFO] Using discovered TorchScript weights: {p}")
            return str(p.resolve())

    raise FileNotFoundError(
        "No valid TorchScript model file found. "
        "Set a file path in weight_path (2.json) or place model.ts/model_gpu.ts/model_cpu.ts in the project folder."
    )


def build_processor(module, cfg: dict, weight_path: str):
    """Create a detector processor across 2.py API variants.

    Supports class names:
      - TorchScriptCameraProcessor
      - TorchScriptProcessor

    Supports constructor signatures:
      - (model_path=..., params=...)
      - (weight_path=..., params=...)
      - positional (path, params)
    """
    params = dict(cfg.get("image_params_1") or cfg.get("camera2_params", {}) or {})

    def _relax_live_params(p: dict, label: str):
        # Live preview should favor visibility over aggressive filtering.
        mask_min_area = float(p.get("mask_min_area", 0) or 0)
        if mask_min_area >= 50000:
            print(f"[INFO] Relaxing {label}.mask_min_area for live preview: {mask_min_area} -> 1200")
            p["mask_min_area"] = 1200

        paste_min_area = float(p.get("paste_area_min_area", 0) or 0)
        if paste_min_area > 5000:
            print(f"[INFO] Relaxing {label}.paste_area_min_area for live preview: {paste_min_area} -> 120")
            p["paste_area_min_area"] = 120

        # Keep live threshold practical if profile is too strict.
        score_thresh = float(p.get("score_thresh", 0.5) or 0.5)
        if score_thresh > 0.6:
            p["score_thresh"] = 0.5

        # Ensure consistency between min/max area gates.
        min_a = float(p.get("mask_min_area", 30) or 30)
        max_a = float(p.get("mask_max_area", 99999999) or 99999999)
        if max_a <= min_a:
            p["mask_max_area"] = max(min_a * 2.0, 99999999)

    _relax_live_params(params, "params")

    # If active profile is used (as in 2.py), relax that nested profile too.
    active_cell_type = str(params.get("active_cell_type", "") or "").strip()
    profiles = params.get("cell_type_profiles", {})
    if active_cell_type and isinstance(profiles, dict) and active_cell_type in profiles:
        profile = profiles.get(active_cell_type)
        if isinstance(profile, dict):
            _relax_live_params(profile, f"cell_type_profiles.{active_cell_type}")

    cls = None
    for name in ("TorchScriptCameraProcessor", "TorchScriptProcessor"):
        if hasattr(module, name):
            cls = getattr(module, name)
            break
    if cls is None:
        raise RuntimeError(
            "No supported processor class found in script. "
            "Expected TorchScriptCameraProcessor or TorchScriptProcessor."
        )

    # Try keyword-based signatures first, then positional fallback.
    for kwargs in (
        {"model_path": weight_path, "params": params},
        {"weight_path": weight_path, "params": params},
    ):
        try:
            return cls(**kwargs)
        except TypeError:
            pass

    return cls(weight_path, params)


def main():
    parser = argparse.ArgumentParser(description="Live camera cell-centering with TorchScript detection")
    parser.add_argument("--source", choices=["auto", "webcam", "flir"], default="auto",
                        help="Capture source: webcam, FLIR bridge, or auto fallback")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index to open directly")
    parser.add_argument("--max-camera-index", type=int, default=9, help="Max webcam index to scan")
    parser.add_argument("--config", type=str, default="2.json", help="Path to model config JSON")
    parser.add_argument("--script", type=str, default="2.py", help="Path to TorchScript processor script")
    parser.add_argument("--flir-python", type=str, default="", help="Path to FLIR PySpin python.exe")
    parser.add_argument("--bridge-script", type=str, default="flir_live_bridge.py", help="Path to FLIR bridge script")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width override")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height override")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (root_dir / script_path).resolve()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root_dir / config_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"TorchScript processor script not found: {script_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    module = load_module_from_path(script_path, "cell_detector_module")
    cfg = module.load_config(config_path)

    flir_preprocess_cfg = dict(cfg.get("flir_preprocess", {}) or {})

    weight_path = resolve_weight_path(cfg, root_dir)
    processor = build_processor(module, cfg=cfg, weight_path=weight_path)

    def _set_centering_state(active: bool, reset_counter: bool = True) -> bool:
        if hasattr(processor, "set_centering_active"):
            processor.set_centering_active(bool(active), reset_counter=reset_counter)
            return bool(getattr(processor, "centering_active", bool(active)))
        # Fallback for older processor variants without helper method.
        processor.centering_active = bool(active)
        if reset_counter and hasattr(processor, "_centered_count"):
            processor._centered_count = 0
        if hasattr(processor, "cell_centered"):
            processor.cell_centered = False
        return bool(getattr(processor, "centering_active", bool(active)))

    ui_centering_active = _set_centering_state(True, reset_counter=True)
    ui_detection_active = True

    cap = None
    bridge_proc = None
    use_flir = False
    active_webcam_index = None

    if args.source == "flir":
        use_flir = True
    elif args.source == "webcam":
        use_flir = False
    else:
        cams = find_working_cameras(max_index=max(0, int(args.max_camera_index)))
        flir_py = _get_flir_python_path(root_dir=root_dir, explicit_path=(args.flir_python or None))
        source_choice = choose_capture_source_auto(
            webcam_available=len(cams) > 0,
            flir_available=bool(flir_py),
        )
        use_flir = source_choice == "flir"
        print(f"[INFO] Selected source: {source_choice}")

    if use_flir:
        bridge_proc = start_flir_bridge(
            root_dir=root_dir,
            flir_python=(args.flir_python or None),
            bridge_script=args.bridge_script,
        )
        apply_flir_camera_settings(bridge_proc, cfg)
    else:
        chosen_idx = args.camera_index
        if chosen_idx is None:
            chosen_idx = select_camera(
                default_index=int(cfg.get("webcam_index", 0)),
                max_index=max(0, int(args.max_camera_index)),
            )
        active_webcam_index = int(chosen_idx)
        cap = open_camera(chosen_idx)

    cam_w = int(args.width or cfg.get("camera_width", 1280))
    cam_h = int(args.height or cfg.get("camera_height", 720))
    if cap is not None:
        cap = configure_webcam_capture(cap, cam_w, cam_h)

    window_name = "Cell Centering Live View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 850)

    prev_t = cv2.getTickCount()
    last_good_frame = None

    def _handle_runtime_key(key: int) -> bool:
        nonlocal ui_centering_active, ui_detection_active
        if key in (ord("p"), ord("P")):
            ui_detection_active = not ui_detection_active
            ui_centering_active = _set_centering_state(not ui_centering_active, reset_counter=True)
            print(
                f"[CELL_CENTERING] Pause toggle -> centering={'ACTIVE' if ui_centering_active else 'PAUSED'}, "
                f"detection={'ACTIVE' if ui_detection_active else 'PAUSED'}"
            )
            return False
        if key in (ord("d"), ord("D")):
            ui_detection_active = not ui_detection_active
            print(f"[CELL_CENTERING] Detection {'resumed' if ui_detection_active else 'paused'} by user.")
            return False
        if key == ord("1"):
            processor.target_mode = "left"
            print("[CELL_CENTERING] Target mode set to LEFT incoming cell.")
            return False
        if key == ord("2"):
            processor.target_mode = "center"
            print("[CELL_CENTERING] Target mode set to CENTER incoming cell.")
            return False
        if key == ord("3"):
            processor.target_mode = "right"
            print("[CELL_CENTERING] Target mode set to RIGHT incoming cell.")
            return False
        return key in (ord("q"), ord("Q"))

    print("[INFO] Controls: p = pause/resume centering, d = pause/resume detection, 1 = left cell, 2 = center cell, 3 = right cell, q = quit")
    bad_frame_count = 0
    max_consecutive_bad = 10
    try:
        while True:
            if use_flir:
                frame = read_flir_frame_bgr(bridge_proc, cfg)
                ret = frame is not None
            else:
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    bad_frame_count += 1
                    print(f"[WARN] Camera read error [{bad_frame_count}/{max_consecutive_bad}]: malformed frame")
                    if bad_frame_count >= max_consecutive_bad:
                        print(f"[WARN] Too many bad frames; attempting camera reset...")
                        try:
                            reset_index = int(
                                active_webcam_index
                                if active_webcam_index is not None
                                else (args.camera_index or int(cfg.get("webcam_index", 0)))
                            )
                            cap = reopen_webcam(cap, reset_index, cam_w, cam_h)
                            bad_frame_count = 0
                        except Exception as re:
                            print(f"[ERROR] Camera reset failed: {re}")
                    key = cv2.waitKey(1) & 0xFF
                    if _handle_runtime_key(key):
                        break
                    continue
                except Exception as e:
                    bad_frame_count += 1
                    print(f"[WARN] Unexpected camera error [{bad_frame_count}/{max_consecutive_bad}]: {e}")
                    key = cv2.waitKey(1) & 0xFF
                    if _handle_runtime_key(key):
                        break
                    continue

            if not ret or frame is None:
                bad_frame_count += 1
                if (not use_flir) and bad_frame_count >= max_consecutive_bad:
                    print(f"[WARN] Too many empty webcam frames; attempting camera reset...")
                    try:
                        reset_index = int(
                            active_webcam_index
                            if active_webcam_index is not None
                            else (args.camera_index or int(cfg.get("webcam_index", 0)))
                        )
                        cap = reopen_webcam(cap, reset_index, cam_w, cam_h)
                        bad_frame_count = 0
                    except Exception as re:
                        print(f"[ERROR] Camera reset failed: {re}")
                if use_flir:
                    if last_good_frame is not None:
                        vis = last_good_frame.copy()
                    else:
                        vis = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                    cv2.putText(
                        vis,
                        "Waiting for FLIR frame...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if _handle_runtime_key(key):
                    break
                continue

            if frame.dtype != np.uint8 or frame.ndim != 3:
                bad_frame_count += 1
                print(f"[WARN] Invalid frame format, skipping")
                if (not use_flir) and bad_frame_count >= max_consecutive_bad:
                    print(f"[WARN] Too many invalid webcam frames; attempting camera reset...")
                    try:
                        reset_index = int(
                            active_webcam_index
                            if active_webcam_index is not None
                            else (args.camera_index or int(cfg.get("webcam_index", 0)))
                        )
                        cap = reopen_webcam(cap, reset_index, cam_w, cam_h)
                        bad_frame_count = 0
                    except Exception as re:
                        print(f"[ERROR] Camera reset failed: {re}")
                key = cv2.waitKey(1) & 0xFF
                if _handle_runtime_key(key):
                    break
                continue

            bad_frame_count = 0
            last_good_frame = frame.copy()

            if use_flir:
                try:
                    frame = preprocess_flir_frame(frame, flir_preprocess_cfg)
                except Exception as e:
                    print(f"[WARN] FLIR preprocessing failed, using raw frame: {e}")

            if ui_detection_active:
                try:
                    vis, det_out = processor.process_frame(frame)
                except Exception as e:
                    print(f"[WARN] Detection processing failed: {e}")
                    vis = frame.copy()
                    cv2.putText(
                        vis,
                        "Detection error - showing raw frame",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if _handle_runtime_key(key):
                        break
                    continue
            else:
                vis = frame.copy()
                det_out = {
                    "detections": [],
                    "raw_det_count": 0,
                    "valid_det_count": 0,
                    "cell_centered": False,
                    "centered_cell_count": int(getattr(processor, "centered_cell_count", 0)),
                    "offset_dx": 0.0,
                    "offset_dy": 0.0,
                    "centered_progress": 0.0,
                    "centering_active": bool(ui_centering_active),
                    "target_mode": str(getattr(processor, "target_mode", "center") or "center"),
                }

            # Support both processor return formats:
            # 1) vis, detections_list
            # 2) vis, info_dict (with info_dict["detections"]) 
            raw_after = 0
            valid_after = 0
            cell_centered = False
            centered_cell_count = 0
            offset_dx = 0.0
            offset_dy = 0.0
            centered_progress = 0.0
            centering_active = bool(ui_centering_active)
            centering_status = "Idle"
            has_active_target = False
            target_mode = "center"
            stage_move_dx = 0.0
            stage_move_dy = 0.0
            if isinstance(det_out, dict):
                detections = det_out.get("detections", []) or []
                raw_after = int(det_out.get("raw_det_count", 0) or 0)
                valid_after = int(det_out.get("valid_det_count", 0) or 0)
                cell_centered = bool(det_out.get("cell_centered", False))
                centered_cell_count = int(det_out.get("centered_cell_count", 0))
                offset_dx = float(det_out.get("offset_dx", 0.0))
                offset_dy = float(det_out.get("offset_dy", 0.0))
                stage_move_dx = float(det_out.get("stage_move_dx", offset_dx))
                stage_move_dy = float(det_out.get("stage_move_dy", offset_dy))
                centered_progress = float(det_out.get("centered_progress", 0.0))
                centering_active = bool(det_out.get("centering_active", ui_centering_active))
                centering_status = str(det_out.get("centering_status", "Idle") or "Idle")
                has_active_target = bool(det_out.get("has_active_target", False))
                target_mode = str(det_out.get("target_mode", "center") or "center")
            elif isinstance(det_out, list):
                detections = det_out
                raw_after = len(det_out)
                valid_after = len(det_out)
            else:
                detections = []

            ui_centering_active = bool(getattr(processor, "centering_active", centering_active))
            centering_active = ui_centering_active

            # Keep only dict-like detections to avoid string/invalid entries.
            detections = [d for d in detections if isinstance(d, dict)]

            # ── Stage control hook ────────────────────────────────────
            # Send offset for current unnumbered target to your stage controller.
            # Replace this placeholder with your actual controller call.
            # Example:
            # stage_controller.move_by_offset(stage_move_dx, stage_move_dy)
            if (
                ui_detection_active
                and centering_active
                and (not cell_centered)
                and has_active_target
                and valid_after > 0
            ):
                _ = (stage_move_dx, stage_move_dy)

            # ── Cell-centred event ────────────────────────────────────
            if cell_centered:
                print(f"[CELL_CENTERING] Cell #{centered_cell_count} confirmed centred — "
                      f"offset=({offset_dx:.1f}, {offset_dy:.1f} px). "
                      f"Total centred so far: {centered_cell_count}. "
                      "Ready for next cell.")
                # ↓ Place your conveyor/stage trigger here, e.g.:
                # trigger_next_cell(centered_cell_count)
            # ──────────────────────────────────────────────────────────

            now_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(now_t - prev_t, 1)
            prev_t = now_t

            # Unified top-left HUD box.
            panel_x = 12
            panel_y = 12
            panel_w = min(340, max(260, vis.shape[1] - panel_x - 12))
            panel_h = 147
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (110, 110, 110), 1)

            center_state = "ACTIVE" if centering_active else "PAUSED"
            detect_state = "ACTIVE" if ui_detection_active else "PAUSED"
            lines = [
                (f"FPS: {fps:.1f}", (0, 255, 255)),
                (f"Offset: dx={int(round(offset_dx))} dy={int(round(offset_dy))}", (0, 255, 255)),
                (f"Detections: raw={int(raw_after)} valid={int(valid_after)}", (255, 255, 0)),
                (f"Centering: {center_state} ({centered_progress:.0%})", (0, 255, 0) if centering_active else (0, 200, 255)),
                (f"Detection: {detect_state}", (0, 255, 0) if ui_detection_active else (0, 200, 255)),
            ]

            y = panel_y + 22
            for text, color in lines:
                cv2.putText(
                    vis,
                    text,
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                y += 25

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if _handle_runtime_key(key):
                break
    finally:
        if cap is not None:
            cap.release()
        stop_flir_bridge(bridge_proc)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
