from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_update_dict(base: dict, updates: dict) -> dict:
    """Recursively merge `updates` into `base` and return a new dict."""
    merged = dict(base or {})
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class TorchScriptCameraProcessor:
    _global_centered_counter = 0

    def _resolve_counter_state_path(self) -> Path:
        raw = str(self.params.get("center_counter_state_path", "config/center_counter_state.json") or "").strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return p

    def _load_persistent_centered_counter(self) -> int:
        try:
            p = Path(self.center_counter_state_path)
            if not p.exists():
                return 0
            data = json.loads(p.read_text(encoding="utf-8"))
            return max(0, int(data.get("centered_cell_count", 0) or 0))
        except Exception:
            return 0

    def _save_persistent_centered_counter(self, value: int) -> None:
        try:
            p = Path(self.center_counter_state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"centered_cell_count": int(max(0, value))}
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to save center counter state: {e}")

    @staticmethod
    def _apply_active_cell_type_profile(params: dict) -> dict:
        p = dict(params or {})
        active_cell_type = p.get("active_cell_type", None)
        profiles = p.get("cell_type_profiles", {})

        if active_cell_type and isinstance(profiles, dict) and active_cell_type in profiles:
            profile = profiles[active_cell_type]
            p = deep_update_dict(p, profile)
            p["_resolved_active_cell_type"] = active_cell_type
        else:
            p["_resolved_active_cell_type"] = None

        return p

    def __init__(self, model_path: str, params: dict):

        # ✅ Center notification state
        self.centered_message_timer = 0
        self.centered_message_duration = 20  # frames (~0.5s–1s depending on FPS)

  
        self.completed_match_radius = 80.0  # pixels (tune 60–120)

        self.completed_memory = 10 


        self.passed_cells = []   # list of {cx, cy}
        self.passed_match_radius = 80.0
        


        # Always start numbering from 0 for each new window/session.
        TorchScriptCameraProcessor._global_centered_counter = 0
        self.prev_detections = []
        self.detect_hold_frames = 5
        self.detect_hold_counter = 0
        self.current_target_id = None
        self.target_mode = "center"   # "left", "center", "right"
        self.row_group_tol_px = 80

        # ── Centering confirmation state ───────────────────────────────
        # How close (px) the cell centre must be to the frame centre
        # to count as "centred" (Euclidean radius).
        self.centered_threshold_px = 15.0
        self.centered_threshold_ratio = 0.08  # NEW (scale-aware)
        self.centered_motion_tol_px = float((params or {}).get("centered_motion_tol_px", 8.0))
        # Number of consecutive frames the cell must stay inside the
        # threshold before it is declared centred.
        self.centered_confirm_frames = 8
        # Consecutive frames spent inside the deadband so far.
        self._centered_count = 0
        # Public flag: True for exactly one frame after confirmation.
        self.cell_centered = False
        # Centering is automatic by default and can be overridden by UI controls.
        self.centering_active = True
        # After confirming, wait this many frames before re-acquiring
        # the next cell (prevents immediately re-locking the same cell
        # as it slowly drifts away).
        self._cooldown_frames = 20
        self._cooldown_count = 0
        # ──────────────────────────────────────────────────────────────

        # ── Centered-cell registry ────────────────────────────────────
        # Sequential counter: increments each time a new cell is centred.
        self.centered_cell_count = int(TorchScriptCameraProcessor._global_centered_counter)
        # Track IDs that have already been confirmed centred.
        self.centered_track_ids = set()
        # Mapping of track ID -> sequential centred cell number.
        self.centered_id_by_track = {}
        # Metadata for numbered cells, keyed by centered cell number.
        # Each entry stores the latest known position/class so numbering
        # can persist even if a track ID changes.
        self.centered_meta_by_id = {}
        # Consecutive-frames-absent counter per numbered ID.
        # When a numbered cell is not matched in a frame its counter increments;
        # once it exceeds centered_absent_max_frames the entry is evicted so
        # the next incoming cell cannot accidentally inherit the old number.
        self._centered_absent = {}          # {center_id: int absent-frame count}
        self.centered_absent_max_frames = int((params or {}).get("centered_absent_max_frames", 20))
        # How many recently numbered cells to keep visible in the HUD.
        self.centered_history_overlay_max = int((params or {}).get("centered_history_overlay_max", 12))
        # ──────────────────────────────────────────────────────────────

        self.model_path = str(model_path)

        raw_params = dict(params or {})
        self.params = self._apply_active_cell_type_profile(raw_params)
        self.active_cell_type = self.params.get("_resolved_active_cell_type", None)
        self.centered_reassoc_tol_px = float(self.params.get("centered_reassoc_tol_px", 120.0))
        self.center_counter_state_path = self._resolve_counter_state_path()

        self.class_names = self.params.get(
            "class_names",
            ["PV cell up", "Paste area", "PV cell down", "Paste area down"]
        )

        self.score_thresh = float(
            self.params.get(
                "score_thresh",
                get_nested(self.params, "detectron2", "score_thresh", default=0.5)
            )
        )
        self.mask_thresh = float(self.params.get("mask_thresh", 0.35))
        self.normalize_input = bool(self.params.get("normalize_input", False))

        device_pref = self.params.get(
            "device",
            get_nested(self.params, "detectron2", "device", default="cuda")
        )
        self.use_cuda = bool(self.params.get("use_cuda", str(device_pref).lower() == "cuda"))
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.debug = bool(self.params.get("debug", False))
        self.debug_every_n = int(self.params.get("debug_every_n", 30))
        self.frame_count = 0
        self.topk_keep = int(self.params.get("topk_keep", 32))
        self.split_pv_components = bool(self.params.get("split_pv_components", False))
        self.split_valley_ratio_max = float(self.params.get("split_valley_ratio_max", 0.90))
        self.split_force_when_elongated = bool(self.params.get("split_force_when_elongated", True))
        self.split_force_aspect_margin = float(self.params.get("split_force_aspect_margin", 0.35))
        self.split_force_valley_ratio_max = float(self.params.get("split_force_valley_ratio_max", 1.12))
        self.target_class_names = list(
            self.params.get("target_class_names", ["PV cell up", "PV cell down"])
        )
        self.target_class_name_set = set(self.target_class_names)

        print("[INFO] torch:", torch.__version__)
        print("[INFO] torchvision:", torchvision.__version__)
        print("[INFO] device:", self.device)
        print("[INFO] loading TorchScript model from:", self.model_path)
        print("[INFO] active_cell_type:", self.active_cell_type)
        print("[INFO] target_class_names:", self.target_class_names)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # self.model = torch.jit.load(self.mdoel_path, map_location=self.device)
        # self.model = self.model.to(self.device)
        # self.model.eval()

        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model = self.model.to(self.device)

        # Some TorchScript models already have training=False as a frozen constant.
        # Calling .eval() on those models can raise:
        # RuntimeError: Can't set constant 'training' which has value:False
        try:
            self.model.eval()
            print("[INFO] model.eval() applied")
        except RuntimeError as e:
            if "Can't set constant 'training'" in str(e):
                print("[INFO] TorchScript model already has training=False constant; skip model.eval()")
            else:
                raise

        try:
            if hasattr(self.model, "training"):
                print(f"[INFO] scripted model training flag = {self.model.training}")
        except Exception:
            pass


        self.roi = self.params.get("roi", None)
        self.pick_roi = self.params.get("pick_roi", self.roi)
        self.place_roi = self.params.get("place_roi", self.roi)
        self.flip_station_roi = self.params.get("flip_station_roi", self.roi)
        self.camera_3_roi = self.params.get("camera_3_roi", None)

        self.axis_method = "pca"
        self.orientation_mode = "pca_repaired_paste_hybrid"

        self.mask_min_area = int(self.params.get("mask_min_area", 30))
        self.mask_max_area = int(self.params.get("mask_max_area", 99999999))
        self.mask_max_area_ratio = float(self.params.get("mask_max_area_ratio", 0.40))

        self.open_kernel = int(self.params.get("open_kernel", 1))
        self.close_kernel = int(self.params.get("close_kernel", 3))
        self.alpha_mask = float(self.params.get("alpha_mask", 0.2))

        self.pixel_gate = float(self.params.get("pixel_gate", 40.0))
        self.max_age = int(self.params.get("max_age", 5))
        self.min_hits = int(self.params.get("min_hits", 1))
        self.stable_required = int(self.params.get("stable_required", 2))
        self.position_jitter_tol = float(self.params.get("position_jitter_tol", 8.0))
        self.angle_jitter_tol = float(self.params.get("angle_jitter_tol", 10.0))
        self.ema_alpha = float(self.params.get("ema_alpha", 0.7))

        self.same_family_overlap_iou = float(self.params.get("same_family_overlap_iou", 0.35))
        self.same_family_overlap_ratio = float(self.params.get("same_family_overlap_ratio", 0.60))

        self.pv_aspect_min = float(self.params.get("pv_aspect_min", 1.35))
        self.pv_aspect_max = float(self.params.get("pv_aspect_max", 2.15))
        self.single_pv_target = bool(self.params.get("single_pv_target", True))
        self.pv_min_area_ratio_to_max = float(
            self.params.get("pv_min_area_ratio_to_max", 0.45)
        )
        self.pv_min_fill_ratio = float(
            self.params.get("pv_min_fill_ratio", 0.45)
        )
        self.pv_down_min_area = float(
            self.params.get("pv_down_min_area", max(12000.0, 2.0 * float(self.mask_min_area)))
        )
        self.pv_down_aspect_min = float(
            self.params.get("pv_down_aspect_min", self.pv_aspect_min)
        )
        self.pv_down_aspect_max = float(
            self.params.get("pv_down_aspect_max", min(self.pv_aspect_max, 2.8))
        )
        self.pv_down_min_fill_ratio = float(
            self.params.get("pv_down_min_fill_ratio", 0.55)
        )

        self.paste_area_min_area = int(self.params.get("paste_area_min_area", 120))
        self.paste_area_max_area = int(self.params.get("paste_area_max_area", 6000))
        self.paste_area_aspect_min = float(self.params.get("paste_area_aspect_min", 1.05))
        self.paste_area_aspect_max = float(self.params.get("paste_area_aspect_max", 2.20))

        self.same_class_center_dist = float(self.params.get("same_class_center_dist", 12.0))
        self.same_class_mask_iou = float(self.params.get("same_class_mask_iou", 0.60))
        self.paste_to_cell_max_center_dist = float(self.params.get("paste_to_cell_max_center_dist", 140.0))
        self.paste_to_cell_min_overlap_keep = float(self.params.get("paste_to_cell_min_overlap_keep", 0.02))

        self.orientation_pair_max_dist = float(self.params.get("orientation_pair_max_dist", 180.0))
        self.orientation_min_proj_abs = float(self.params.get("orientation_min_proj_abs", 4.0))

        self.end_region_frac = float(self.params.get("end_region_frac", 0.22))
        self.end_shape_min_ratio_diff = float(self.params.get("end_shape_min_ratio_diff", 0.04))
        self.end_shape_vote_weight = float(self.params.get("end_shape_vote_weight", 0.65))

        self.orientation_subtract_paste = bool(self.params.get("orientation_subtract_paste", True))
        self.orientation_paste_dilate = int(self.params.get("orientation_paste_dilate", 3))

        self.repair_use_convex_hull = bool(self.params.get("repair_use_convex_hull", True))
        self.repair_close_kernel = int(self.params.get("repair_close_kernel", 5))
        self.repair_open_kernel = int(self.params.get("repair_open_kernel", 1))
        self.repair_min_component_area = int(self.params.get("repair_min_component_area", 80))
        self.repair_fill_minrect = bool(self.params.get("repair_fill_minrect", True))
        self.repair_fill_minrect_extent_thresh = float(
            self.params.get("repair_fill_minrect_extent_thresh", 0.72)
        )
        self.repair_fill_minrect_aspect_tol = float(
            self.params.get("repair_fill_minrect_aspect_tol", 0.45)
        )

        self.side_n_bins = int(self.params.get("side_n_bins", 25))
        self.side_score_min = float(self.params.get("side_score_min", 0.035))
        self.side_vote_weight = float(self.params.get("side_vote_weight", 1.10))
        self.side_area_weight = float(self.params.get("side_area_weight", 0.20))
        self.side_width_weight = float(self.params.get("side_width_weight", 0.80))
        self.side_width_quantile = float(self.params.get("side_width_quantile", 0.80))
        self.side_flip_mapping = bool(self.params.get("side_flip_mapping", False))

        self.use_paste_side_cue = bool(self.params.get("use_paste_side_cue", True))
        self.paste_side_vote_weight = float(self.params.get("paste_side_vote_weight", 1.20))
        self.paste_side_score_min = float(self.params.get("paste_side_score_min", 0.02))
        self.paste_count_weight = float(self.params.get("paste_count_weight", 0.25))
        self.paste_area_weight = float(self.params.get("paste_area_weight", 0.55))
        self.paste_proj_weight = float(self.params.get("paste_proj_weight", 0.20))
        self.paste_side_flip_mapping = bool(self.params.get("paste_side_flip_mapping", False))

        self.use_paste_as_weak_cue = bool(self.params.get("use_paste_as_weak_cue", False))
        self.paste_vote_weight = float(self.params.get("paste_vote_weight", 0.20))

        self.orientation_memory_max_dist = float(self.params.get("orientation_memory_max_dist", 80.0))
        self.orientation_flip_guard_deg = float(self.params.get("orientation_flip_guard_deg", 120.0))
        self.orientation_keep_prev_margin = float(self.params.get("orientation_keep_prev_margin", 0.18))

        self.next_track_id = 1
        self.tracks: Dict[int, dict] = {}

        print(f"[INFO] pick_roi = {self.pick_roi}")
        print(f"[INFO] place_roi = {self.place_roi}")
        print(f"[INFO] flip_station_roi = {self.flip_station_roi}")


        
        self.current_target_id = None
        self.target_mode = "center"   # "left", "center", "right"

        # Row/column settings
        self.row_group_tol_px = 80    # how close vertically = same row


    def set_centering_active(self, active: bool, reset_counter: bool = True) -> None:
        self.centering_active = bool(active)
        self.cell_centered = False
        if reset_counter:
            self._centered_count = 0




    def group_by_rows(self, detections):
        if not detections:
            return []

        detections_sorted = sorted(detections, key=lambda d: d["cy"])
        rows = []

        for det in detections_sorted:
            placed = False
            for row in rows:
                mean_y = sum(d["cy"] for d in row) / len(row)
                if abs(det["cy"] - mean_y) < self.row_group_tol_px:
                    row.append(det)
                    placed = True
                    break

            if not placed:
                rows.append([det])

        return rows



    def select_target_from_rows(self, detections, center_x, center_y):
        rows = self.group_by_rows(detections)

        if not rows:
            return None

        # choose row closest to center
        target_row = min(
            rows,
            key=lambda row: abs(
                (sum(d["cy"] for d in row) / len(row)) - center_y
            )
        )

        target_row = sorted(target_row, key=lambda d: d["cx"])

        if self.target_mode == "left":
            return target_row[0]
        elif self.target_mode == "right":
            return target_row[-1]
        elif self.target_mode == "center":
            return target_row[len(target_row) // 2]

        return target_row[0]


    def _filter_already_centered(self, detections):
        """Return only detections that are NOT already confirmed centered by track ID."""
        if not self.centered_track_ids:
            return detections
        out = []
        for d in detections:
            tid = d.get("_track_id", None)
            if tid is None or int(tid) not in self.centered_track_ids:
                out.append(d)
        return out

    def _reassociate_centered_tracks(self, detections):
        """Update metadata for already-mapped centered tracks.

        Important safety rule:
        Do not assign a centered ID to a brand-new track by proximity. That can
        cause a new incoming cell to inherit an old number (e.g., #1 reused).
        """
        if not self.centered_meta_by_id:
            return

        # IDs already represented by mapped tracks this frame cannot be reused.
        occupied_ids = set()
        for d in detections:
            tid = d.get("_track_id", None)
            if tid is None:
                continue
            cid = self.centered_id_by_track.get(int(tid), None)
            if cid is not None:
                occupied_ids.add(int(cid))

        for d in detections:
            tid = d.get("_track_id", None)
            if tid is None:
                continue
            tid = int(tid)

            # Already mapped to a centered ID.
            if tid in self.centered_id_by_track:
                cid = int(self.centered_id_by_track[tid])
                meta = self.centered_meta_by_id.get(cid, None)
                if meta is not None:
                    meta["cx"] = float(d.get("cx", meta.get("cx", 0.0)))
                    meta["cy"] = float(d.get("cy", meta.get("cy", 0.0)))
                occupied_ids.add(int(cid))


    def get_locked_target(self, detections, center_x, center_y):
        if self.current_target_id is not None:
            for d in detections:
                if d["_track_id"] == self.current_target_id:
                    return d

            # target lost
            self.current_target_id = None

        target = self.select_target_from_rows(detections, center_x, center_y)

        if target is not None:
            self.current_target_id = target.get("_track_id", None)

        return target




    @staticmethod
    def angle_wrap_360(angle: float) -> float:
        angle = float(angle) % 360.0
        if angle < 0:
            angle += 360.0
        return angle

    @staticmethod
    def angle_wrap_180(angle: float) -> float:
        angle = float(angle) % 180.0
        if angle < 0:
            angle += 180.0
        return angle

    @staticmethod
    def angle_diff_deg_360(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    @staticmethod
    def shortest_signed_delta_360(a: float, b: float) -> float:
        return (b - a + 180.0) % 360.0 - 180.0

    def smooth_angle_360(self, prev_angle: float, new_angle: float) -> float:
        prev = self.angle_wrap_360(prev_angle)
        new = self.angle_wrap_360(new_angle)
        d = self.shortest_signed_delta_360(prev, new)
        out = prev + self.ema_alpha * d
        return self.angle_wrap_360(out)

    def draw_mask_overlay(self, image_bgr, binary_mask, alpha=0.35):
        overlay = image_bgr.copy()
        overlay[binary_mask > 0] = (0, 255, 255)
        return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    def _get_roi_by_mode(self, roi_mode: str):
        roi_mode = str(roi_mode).lower()

        if roi_mode == "pick":
            return self.pick_roi

        if roi_mode == "place":
            return self.place_roi

        if roi_mode == "flip_station":
            return self.flip_station_roi

        if roi_mode in ("camera_3", "camera3", "welding"):
            return self.camera_3_roi   # None means full camera3 frame

        return self.roi

    def preprocess_roi(self, frame: np.ndarray, roi_mode: str = "default"):
        roi = self._get_roi_by_mode(roi_mode)

        if roi is None:
            return frame, frame, (0, 0), None

        x1, y1, x2, y2 = map(int, roi)
        h, w = frame.shape[:2]

        if not (0 <= x1 < w and 0 <= x2 <= w and 0 <= y1 < h and 0 <= y2 <= h):
            raise ValueError(f"ROI out of bounds: {roi}, frame size={(w, h)}")

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid ROI [x1, y1, x2, y2]: {roi}")

        roi_frame = frame[y1:y2, x1:x2].copy()
        return frame, roi_frame, (x1, y1), (x1, y1, x2, y2)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)

        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

        if self.close_kernel > 1:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        return m

    def _prepare_input_tensor(self, infer_frame: np.ndarray) -> torch.Tensor:
        x = np.ascontiguousarray(infer_frame)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous().float()
        if self.normalize_input:
            x = x / 255.0
        return x.to(self.device)

    def paste_mask_to_image(self, mask_prob, box, image_h, image_w, thresh=0.5):
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w))
        y2 = max(0, min(y2, image_h))

        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)

        mask_prob = np.asarray(mask_prob, dtype=np.float32)
        resized_mask = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (resized_mask > thresh).astype(np.uint8)

        full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask[: y2 - y1, : x2 - x1]
        return full_mask

    def _mask_to_full_image(self, mask_raw, box, image_h, image_w, thresh=0.5):
        mask = np.asarray(mask_raw)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask = mask.astype(np.float32)

        if mask.shape[0] == image_h and mask.shape[1] == image_w:
            return (mask > thresh).astype(np.uint8)

        return self.paste_mask_to_image(mask, box, image_h, image_w, thresh=thresh)

    def _largest_contour_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_u8

        largest = max(contours, key=cv2.contourArea)
        out = np.zeros_like(mask_u8)
        cv2.drawContours(out, [largest], -1, 1, thickness=-1)
        return out

    def _filter_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        out = np.zeros_like(mask_u8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                out[labels == i] = 1
        return out

    def _compute_pca_axis_from_mask(self, full_mask: np.ndarray):
        ys, xs = np.nonzero(full_mask > 0)
        if len(xs) < 10:
            return None

        pts = np.column_stack([xs, ys]).astype(np.float32)
        mean = np.mean(pts, axis=0)
        pts0 = pts - mean

        cov = np.cov(pts0.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major = eigvecs[:, 0]
        minor = eigvecs[:, 1]

        vx, vy = float(major[0]), float(major[1])
        axis_angle = self.angle_wrap_360(math.degrees(math.atan2(vy, vx)))

        lam0 = float(max(eigvals[0], 1e-9))
        lam1 = float(max(eigvals[1], 1e-9))
        aspect_ratio = float(math.sqrt(lam0 / lam1))

        return {
            "cx_pca": float(mean[0]),
            "cy_pca": float(mean[1]),
            "axis_angle_pca": axis_angle,
            "major_vec": (vx, vy),
            "minor_vec": (float(minor[0]), float(minor[1])),
            "aspect_ratio_pca": aspect_ratio,
        }

    def mask_to_geometry(self, full_mask: np.ndarray, name: Optional[str] = None):
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.mask_min_area:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx_m = float(M["m10"] / M["m00"])
        cy_m = float(M["m01"] / M["m00"])

        pca_geom = self._compute_pca_axis_from_mask(full_mask)
        if pca_geom is None:
            return None

        axis_angle_pca = float(pca_geom["axis_angle_pca"])
        aspect_ratio_pca = float(pca_geom["aspect_ratio_pca"])
        cx_pca = float(pca_geom["cx_pca"])
        cy_pca = float(pca_geom["cy_pca"])

        rect = cv2.minAreaRect(largest)
        (rect_cx, rect_cy), (rect_w, rect_h), _ = rect
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        rect_edges = []
        for i in range(4):
            p1 = box_pts[i].astype(np.float32)
            p2 = box_pts[(i + 1) % 4].astype(np.float32)
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = float(math.hypot(dx, dy))
            rect_edges.append((length, dx, dy))

        rect_edges.sort(key=lambda x: x[0], reverse=True)
        _, long_dx, long_dy = rect_edges[0]
        axis_angle_rect = self.angle_wrap_360(math.degrees(math.atan2(long_dy, long_dx)))

        rect_long = float(max(rect_w, rect_h))
        rect_short = float(min(rect_w, rect_h))
        aspect_ratio_rect = float(rect_long / max(rect_short, 1e-6))

        if name in ("PV cell up", "PV cell down"):
            cx = float(rect_cx)
            cy = float(rect_cy)
            axis_angle = float(axis_angle_rect)
            aspect_ratio = float(aspect_ratio_rect)
        else:
            cx = cx_m
            cy = cy_m
            axis_angle = axis_angle_pca
            aspect_ratio = aspect_ratio_pca

        return {
            "contour": largest,
            "area": float(area),

            "cx": float(cx),
            "cy": float(cy),

            "cx_moment": float(cx_m),
            "cy_moment": float(cy_m),

            "cx_pca": float(cx_pca),
            "cy_pca": float(cy_pca),

            "cx_rect": float(rect_cx),
            "cy_rect": float(rect_cy),

            "rect_w": float(rect_w),
            "rect_h": float(rect_h),
            "rect_pts": box_pts,

            "axis_angle": float(axis_angle),
            "axis_angle_pca": float(axis_angle_pca),
            "axis_angle_rect": float(axis_angle_rect),

            "aspect_ratio": float(aspect_ratio),
            "aspect_ratio_pca": float(aspect_ratio_pca),
            "aspect_ratio_rect": float(aspect_ratio_rect),
        }

    def _validate_outputs(self, outputs, image_h: int, image_w: int):

        if not isinstance(outputs, (tuple, list)):
            raise TypeError(f"TorchScript scripted output must be tuple/list, got {type(outputs)}")

        if len(outputs) == 0:
            empty_boxes = torch.empty((0, 4), dtype=torch.float32)
            empty_classes = torch.empty((0,), dtype=torch.int64)
            empty_masks = torch.empty((0, image_h, image_w), dtype=torch.float32)
            empty_scores = torch.empty((0,), dtype=torch.float32)
            return empty_boxes, empty_classes, empty_masks, empty_scores

        first = outputs[0]

        if not isinstance(first, dict):
            raise TypeError(f"Expected first output item to be dict, got {type(first)}")

        if "pred_boxes" in first:
            boxes = first["pred_boxes"]
        elif "proposal_boxes" in first:
            boxes = first["proposal_boxes"]
        else:
            raise KeyError(f"Output dict missing 'pred_boxes' and 'proposal_boxes'. Keys={list(first.keys())}")

        if "scores" not in first:
            raise KeyError(f"Output dict missing 'scores'. Keys={list(first.keys())}")
        scores = first["scores"]

        if "pred_classes" not in first:
            raise KeyError(f"Output dict missing 'pred_classes'. Keys={list(first.keys())}")
        classes = first["pred_classes"]

        if "pred_masks" in first:
            masks = first["pred_masks"]
        else:
            n = int(scores.shape[0]) if torch.is_tensor(scores) else 0
            masks = torch.empty((n, image_h, image_w), dtype=torch.float32, device=boxes.device if torch.is_tensor(boxes) else self.device)

        if not torch.is_tensor(boxes):
            raise TypeError(f"boxes must be tensor, got {type(boxes)}")
        if not torch.is_tensor(classes):
            raise TypeError(f"classes must be tensor, got {type(classes)}")
        if not torch.is_tensor(scores):
            raise TypeError(f"scores must be tensor, got {type(scores)}")
        if not torch.is_tensor(masks):
            raise TypeError(f"masks must be tensor, got {type(masks)}")

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes shape must be [N, 4], got {tuple(boxes.shape)}")

        if classes.ndim != 1:
            classes = classes.reshape(-1)

        if scores.ndim != 1:
            scores = scores.reshape(-1)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0)
        elif masks.ndim != 3:
            raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")

        return boxes, classes, masks, scores
    
    @staticmethod
    def _mask_area(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask > 0))

    @staticmethod
    def _intersection_area(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
        return int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = self._intersection_area(mask_a, mask_b)
        if inter <= 0:
            return 0.0
        union = self._mask_area(mask_a) + self._mask_area(mask_b) - inter
        return float(inter / max(union, 1))

    def _overlap_ratio(self, child_mask: np.ndarray, parent_mask: np.ndarray) -> float:
        child_area = self._mask_area(child_mask)
        if child_area <= 0:
            return 0.0
        inter = self._intersection_area(child_mask, parent_mask)
        return float(inter / child_area)

    @staticmethod
    def _center_distance(d1: dict, d2: dict) -> float:
        return float(np.hypot(d1["cx"] - d2["cx"], d1["cy"] - d2["cy"]))

    def _filter_large_masks(self, detections: List[dict], frame_hw: Tuple[int, int]) -> List[dict]:
        H, W = frame_hw
        max_area_by_ratio = int(H * W * self.mask_max_area_ratio)
        out = []
        for d in detections:
            area = int(d["area"])
            if area > self.mask_max_area:
                continue
            if area > max_area_by_ratio:
                continue
            out.append(d)
        return out

    def _filter_pv_by_aspect_ratio(self, detections: List[dict]) -> List[dict]:
        out = []

        for d in detections:
            if d["name"] in ("PV cell up", "PV cell down"):
                ar = float(d.get("aspect_ratio", 0.0))
                area = float(d.get("area", 0.0))

                if d["name"] == "PV cell down":
                    min_ar = float(self.pv_down_aspect_min)
                    max_ar = float(self.pv_down_aspect_max)

                    if area < float(self.pv_down_min_area):
                        if self.debug:
                            print(
                                f"[PV FILTER REJECT] name={d['name']} "
                                f"reason=area area={area:.1f} "
                                f"min={self.pv_down_min_area:.1f}"
                            )
                        continue

                    if not (min_ar <= ar <= max_ar):
                        if self.debug:
                            print(
                                f"[PV FILTER REJECT] name={d['name']} "
                                f"reason=aspect ar={ar:.3f} "
                                f"allowed=[{min_ar:.3f}, {max_ar:.3f}]"
                            )
                        continue

                    rw = float(d.get("rect_w", 0.0) or 0.0)
                    rh = float(d.get("rect_h", 0.0) or 0.0)
                    rect_area = rw * rh
                    fill_ratio = float(area / max(rect_area, 1e-6))
                    if fill_ratio < float(self.pv_down_min_fill_ratio):
                        if self.debug:
                            print(
                                f"[PV FILTER REJECT] name={d['name']} "
                                f"reason=fill_ratio fill={fill_ratio:.3f} "
                                f"min={self.pv_down_min_fill_ratio:.3f}"
                            )
                        continue

                else:
                    if not (self.pv_aspect_min <= ar <= self.pv_aspect_max):
                        if self.debug:
                            print(
                                f"[PV FILTER REJECT] name={d['name']} "
                                f"reason=aspect ar={ar:.3f} "
                                f"allowed=[{self.pv_aspect_min:.3f}, {self.pv_aspect_max:.3f}]"
                            )
                        continue

            out.append(d)

        return out

    def _keep_body_like_pv(self, detections: List[dict]) -> List[dict]:
        """Suppress small/skinny PV fragments (e.g., interconnector-like masks)."""
        if not detections:
            return detections

        pv_idxs = [
            i for i, d in enumerate(detections)
            if d.get("name") in ("PV cell up", "PV cell down")
        ]
        if not pv_idxs:
            return detections

        pv_areas = [float(detections[i].get("area", 0.0) or 0.0) for i in pv_idxs]
        max_area = max(pv_areas) if pv_areas else 0.0
        min_area_dyn = max(float(self.mask_min_area), float(self.pv_min_area_ratio_to_max) * max_area)

        kept_flags = [True] * len(detections)
        pv_kept = []

        for i in pv_idxs:
            d = detections[i]
            area = float(d.get("area", 0.0) or 0.0)
            rw = float(d.get("rect_w", 0.0) or 0.0)
            rh = float(d.get("rect_h", 0.0) or 0.0)
            rect_area = max(rw * rh, 1e-6)
            fill_ratio = float(area / rect_area)

            if area < min_area_dyn or fill_ratio < float(self.pv_min_fill_ratio):
                kept_flags[i] = False
                continue
            pv_kept.append(i)

        # In live centering mode, keep only one strongest PV body-like target.
        if self.single_pv_target and pv_kept:
            best_i = max(
                pv_kept,
                key=lambda j: (
                    float(detections[j].get("area", 0.0)),
                    float(detections[j].get("score", 0.0)),
                ),
            )
            for i in pv_kept:
                kept_flags[i] = (i == best_i)

        return [d for d, k in zip(detections, kept_flags) if k]
    
    def _split_mask_into_components(
        self,
        full_mask: np.ndarray,
        name: str,
        score: float,
        min_component_area: int = 800
    ) -> List[np.ndarray]:
        mask_u8 = (full_mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        parts = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(min_component_area):
                continue

            comp = np.zeros_like(mask_u8, dtype=np.uint8)
            comp[labels == i] = 1
            parts.append(comp)

        if len(parts) > 1:
            print(
                f"[MASK SPLIT] name={name}, score={score:.3f}, "
                f"split_into={len(parts)} connected components"
            )
            return parts

        if name not in ("PV cell up", "PV cell down"):
            return [mask_u8]

        ys, xs = np.nonzero(mask_u8)
        if len(xs) < int(2 * min_component_area):
            return [mask_u8]

        pts = np.column_stack([xs, ys]).astype(np.float32)
        mean = np.mean(pts, axis=0)
        pts0 = pts - mean

        cov = np.cov(pts0.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        major = eigvecs[:, order[0]]
        minor = eigvecs[:, order[1]]

        ux, uy = float(major[0]), float(major[1])
        vx, vy = float(minor[0]), float(minor[1])

        u = pts0[:, 0] * ux + pts0[:, 1] * uy
        v = pts0[:, 0] * vx + pts0[:, 1] * vy

        u_min = float(np.min(u))
        u_max = float(np.max(u))
        span_u = u_max - u_min

        if span_u < 20.0:
            return [mask_u8]

        n_bins = max(40, int(span_u / 6))
        bin_edges = np.linspace(u_min, u_max, n_bins + 1)

        width_profile = []
        center_profile = []

        for i in range(n_bins):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            m = (u >= left) & (u < right)
            if np.count_nonzero(m) < 8:
                width_profile.append(0.0)
                center_profile.append(0.5 * (left + right))
                continue

            vv = v[m]
            width = float(np.max(vv) - np.min(vv))
            width_profile.append(width)
            center_profile.append(0.5 * (left + right))

        width_profile = np.asarray(width_profile, dtype=np.float32)
        center_profile = np.asarray(center_profile, dtype=np.float32)

        if len(width_profile) < 5:
            return [mask_u8]

        smooth = width_profile.copy()
        if len(smooth) >= 5:
            smooth = np.convolve(smooth, np.ones(5, dtype=np.float32) / 5.0, mode="same")

        search_l = int(0.20 * len(smooth))
        search_r = int(0.80 * len(smooth))
        if search_r - search_l < 3:
            return [mask_u8]

        inner = smooth[search_l:search_r]
        cut_rel = int(np.argmin(inner))
        cut_idx = search_l + cut_rel
        cut_u = float(center_profile[cut_idx])

        edge_ref = float(max(np.median(smooth[:max(3, search_l)]), np.median(smooth[min(len(smooth)-3, search_r):])))
        valley = float(smooth[cut_idx])

        if edge_ref <= 1e-6:
            return [mask_u8]

        valley_ratio = valley / edge_ref

        if valley_ratio > float(self.split_valley_ratio_max):
            force_split = False
            if self.split_force_when_elongated:
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest)
                    (_rcx, _rcy), (rw, rh), _rang = rect
                    rect_long = float(max(rw, rh))
                    rect_short = float(max(min(rw, rh), 1e-6))
                    rect_aspect = float(rect_long / rect_short)
                    force_aspect_thr = float(self.pv_aspect_max) + float(self.split_force_aspect_margin)

                    if (
                        rect_aspect >= force_aspect_thr
                        and valley_ratio <= float(self.split_force_valley_ratio_max)
                    ):
                        force_split = True
                        print(
                            f"[MASK SPLIT] name={name}, score={score:.3f}, "
                            f"force split on elongated mask: valley_ratio={valley_ratio:.3f}, "
                            f"rect_aspect={rect_aspect:.3f}, thr={force_aspect_thr:.3f}"
                        )

            if not force_split:
                print(
                    f"[MASK SPLIT] name={name}, score={score:.3f}, "
                    f"skip split because valley not deep enough: valley_ratio={valley_ratio:.3f}"
                )
                return [mask_u8]

        H, W = mask_u8.shape
        yy, xx = np.indices((H, W))
        xx0 = xx.astype(np.float32) - float(mean[0])
        yy0 = yy.astype(np.float32) - float(mean[1])
        uu = xx0 * ux + yy0 * uy

        part1 = ((mask_u8 > 0) & (uu <= cut_u)).astype(np.uint8)
        part2 = ((mask_u8 > 0) & (uu > cut_u)).astype(np.uint8)

        out_parts = []
        for p in (part1, part2):
            p = self._filter_small_components(p, min_component_area)
            p = self._largest_contour_mask(p)
            area = int(np.count_nonzero(p))
            if area >= int(min_component_area):
                out_parts.append(p)

        if len(out_parts) >= 2:
            print(
                f"[MASK SPLIT] name={name}, score={score:.3f}, "
                f"split merged PV mask into {len(out_parts)} parts by narrow-waist cut, "
                f"valley_ratio={valley_ratio:.3f}"
            )
            return out_parts

        return [mask_u8]
    
    def _deduplicate_same_region_cells(self, detections: List[dict]) -> List[dict]:
        if not detections:
            return detections

        kept = [True] * len(detections)
        idxs = [i for i, d in enumerate(detections) if d["name"] in ("PV cell up", "PV cell down")]

        for a in range(len(idxs)):
            i = idxs[a]
            if not kept[i]:
                continue
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                if not kept[j]:
                    continue

                di = detections[i]
                dj = detections[j]

                iou = self._mask_iou(di["mask"], dj["mask"])
                ri = self._overlap_ratio(di["mask"], dj["mask"])
                rj = self._overlap_ratio(dj["mask"], di["mask"])

                if iou >= self.same_family_overlap_iou or ri >= self.same_family_overlap_ratio or rj >= self.same_family_overlap_ratio:
                    key_i = (float(di["score"]), float(di["area"]))
                    key_j = (float(dj["score"]), float(dj["area"]))
                    if key_i >= key_j:
                        kept[j] = False
                    else:
                        kept[i] = False
                        break

        return [d for d, k in zip(detections, kept) if k]

    def _deduplicate_same_class_small_parts(self, detections: List[dict], class_name: str) -> List[dict]:
        idxs = [i for i, d in enumerate(detections) if d["name"] == class_name]
        if len(idxs) <= 1:
            return detections

        kept = [True] * len(detections)

        for a in range(len(idxs)):
            i = idxs[a]
            if not kept[i]:
                continue
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                if not kept[j]:
                    continue

                di = detections[i]
                dj = detections[j]

                iou = self._mask_iou(di["mask"], dj["mask"])
                dist = self._center_distance(di, dj)

                if iou >= self.same_class_mask_iou and dist <= self.same_class_center_dist:
                    key_i = (float(di["score"]), float(di["area"]))
                    key_j = (float(dj["score"]), float(dj["area"]))
                    if key_i >= key_j:
                        kept[j] = False
                    else:
                        kept[i] = False
                        break

        return [d for d, k in zip(detections, kept) if k]

    def _filter_isolated_paste_candidates(self, detections: List[dict]) -> List[dict]:
        if not detections:
            return detections

        pv_cells = [d for d in detections if d["name"] in ("PV cell up", "PV cell down")]
        if not pv_cells:
            return detections

        out = []
        for d in detections:
            if d["name"] not in ("Paste area", "Paste area down"):
                out.append(d)
                continue

            best_dist = float("inf")
            best_overlap = 0.0

            for cell in pv_cells:
                dist = self._center_distance(d, cell)
                ov = self._overlap_ratio(d["mask"], cell["mask"])
                if dist < best_dist:
                    best_dist = dist
                if ov > best_overlap:
                    best_overlap = ov

            if best_overlap < self.paste_to_cell_min_overlap_keep and best_dist > self.paste_to_cell_max_center_dist:
                continue

            out.append(d)

        return out

    def _expected_paste_name_for_cell(self, cell_name: str) -> Optional[str]:
        if cell_name == "PV cell up":
            return "Paste area"
        if cell_name == "PV cell down":
            return "Paste area down"
        return None

    def _collect_overlap_paste_masks(self, cell_det: dict, detections: List[dict]) -> List[np.ndarray]:
        masks = []
        for d in detections:
            if d["name"] not in ("Paste area", "Paste area down"):
                continue

            overlap = self._overlap_ratio(d["mask"], cell_det["mask"])
            if overlap <= 0.0:
                continue

            pm = (d["mask"] > 0).astype(np.uint8)
            if self.orientation_paste_dilate > 1:
                k = np.ones((self.orientation_paste_dilate, self.orientation_paste_dilate), np.uint8)
                pm = cv2.dilate(pm, k, iterations=1)

            inter = ((pm > 0) & (cell_det["mask"] > 0)).astype(np.uint8)
            if np.count_nonzero(inter) > 0:
                masks.append(inter)
        return masks

    def _make_cell_minus_paste_mask(self, cell_det: dict, detections: List[dict]) -> np.ndarray:
        base = (cell_det["mask"] > 0).astype(np.uint8)
        if not self.orientation_subtract_paste:
            return base

        overlap_masks = self._collect_overlap_paste_masks(cell_det, detections)
        if not overlap_masks:
            return base

        subtract = np.zeros_like(base, dtype=np.uint8)
        for m in overlap_masks:
            subtract = np.maximum(subtract, m)

        out = ((base > 0) & (subtract == 0)).astype(np.uint8)
        return out

    def _repair_orientation_mask(self, minus_paste_mask: np.ndarray) -> np.ndarray:
        m = (minus_paste_mask > 0).astype(np.uint8)

        if self.repair_open_kernel > 1:
            k = np.ones((self.repair_open_kernel, self.repair_open_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

        if self.repair_close_kernel > 1:
            k = np.ones((self.repair_close_kernel, self.repair_close_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        m = self._filter_small_components(m, self.repair_min_component_area)
        m = self._largest_contour_mask(m)

        if self.repair_use_convex_hull:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest)
                hull_mask = np.zeros_like(m)
                cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
                m = hull_mask

        return m.astype(np.uint8)

    def _make_orientation_masks(self, cell_det: dict, detections: List[dict]) -> dict:
        raw_cell_mask = (cell_det["mask"] > 0).astype(np.uint8)

        if cell_det["name"] == "PV cell down":
            minus_paste_mask = raw_cell_mask.copy()
            repaired_mask = self._repair_pv_cell_mask(raw_cell_mask)
        else:
            minus_paste_mask = self._make_cell_minus_paste_mask(cell_det, detections)
            repaired_mask = self._repair_orientation_mask(minus_paste_mask)

        return {
            "raw_cell_mask": raw_cell_mask,
            "minus_paste_mask": minus_paste_mask,
            "repaired_mask": repaired_mask,
        }
    
    def _repair_pv_cell_mask(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)

        if self.open_kernel > 1:
            k = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

        strong_close_kernel = max(self.repair_close_kernel, 9)
        if strong_close_kernel > 1:
            k = np.ones((strong_close_kernel, strong_close_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        m = self._filter_small_components(m, min(30, self.repair_min_component_area))
        m = self._largest_contour_mask(m)

        if self.repair_use_convex_hull:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest)
                hull_mask = np.zeros_like(m)
                cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
                m = hull_mask

        if self.repair_fill_minrect:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(largest))
                rect = cv2.minAreaRect(largest)
                (_cx, _cy), (rw, rh), _ = rect
                rect_area = float(max(rw * rh, 1e-6))
                extent = float(area / rect_area)

                rect_long = float(max(rw, rh))
                rect_short = float(max(min(rw, rh), 1e-6))
                rect_aspect = float(rect_long / rect_short)

                min_ar = max(1.01, float(self.pv_aspect_min) - float(self.repair_fill_minrect_aspect_tol))
                max_ar = float(self.pv_aspect_max) + float(self.repair_fill_minrect_aspect_tol)

                if extent < float(self.repair_fill_minrect_extent_thresh) and (min_ar <= rect_aspect <= max_ar):
                    box_pts = cv2.boxPoints(rect).astype(np.int32)
                    rect_mask = np.zeros_like(m)
                    cv2.drawContours(rect_mask, [box_pts], -1, 1, thickness=-1)
                    m = np.maximum(m, rect_mask)

        return m.astype(np.uint8)

    def _compute_axis_from_repaired_mask(self, repaired_mask: np.ndarray) -> dict:
        mask_u8 = (repaired_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "valid": False,
                "cx": 0.0,
                "cy": 0.0,
                "axis_angle": 0.0,
                "aspect_ratio": 0.0,
            }

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < max(20.0, float(self.mask_min_area)):
            return {
                "valid": False,
                "cx": 0.0,
                "cy": 0.0,
                "axis_angle": 0.0,
                "aspect_ratio": 0.0,
            }

        rect = cv2.minAreaRect(largest)
        (rect_cx, rect_cy), (rect_w, rect_h), _ = rect
        box_pts = cv2.boxPoints(rect).astype(np.int32)

        rect_edges = []
        for i in range(4):
            p1 = box_pts[i].astype(np.float32)
            p2 = box_pts[(i + 1) % 4].astype(np.float32)
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = float(math.hypot(dx, dy))
            rect_edges.append((length, dx, dy))

        rect_edges.sort(key=lambda x: x[0], reverse=True)
        _, long_dx, long_dy = rect_edges[0]
        axis_angle_rect = self.angle_wrap_360(math.degrees(math.atan2(long_dy, long_dx)))

        rect_long = float(max(rect_w, rect_h))
        rect_short = float(min(rect_w, rect_h))
        aspect_ratio_rect = float(rect_long / max(rect_short, 1e-6))

        return {
            "valid": True,
            "cx": float(rect_cx),
            "cy": float(rect_cy),
            "axis_angle": float(axis_angle_rect),
            "aspect_ratio": float(aspect_ratio_rect),
        }
    
    def _compute_end_shape_signature(self, axis_angle: float, cx: float, cy: float, ref_mask: np.ndarray) -> dict:
        ys, xs = np.nonzero(ref_mask > 0)

        if len(xs) == 0:
            return {
                "pos_count": 0,
                "neg_count": 0,
                "ratio_diff": 0.0,
                "suggested_angle360": self.angle_wrap_360(axis_angle),
                "valid": False,
            }

        theta = math.radians(axis_angle)
        ux = math.cos(theta)
        uy = math.sin(theta)

        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        proj = dx * ux + dy * uy

        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        span = max(pmax - pmin, 1e-6)
        end_len = span * self.end_region_frac

        pos_region = proj >= (pmax - end_len)
        neg_region = proj <= (pmin + end_len)

        pos_count = int(np.count_nonzero(pos_region))
        neg_count = int(np.count_nonzero(neg_region))
        total = max(pos_count + neg_count, 1)

        ratio_diff = float(abs(pos_count - neg_count) / total)

        if pos_count <= neg_count:
            suggested = axis_angle
        else:
            suggested = axis_angle + 180.0

        return {
            "pos_count": pos_count,
            "neg_count": neg_count,
            "ratio_diff": ratio_diff,
            "suggested_angle360": self.angle_wrap_360(suggested),
            "valid": True,
        }

    def _robust_side_width(self, widths: List[float]) -> float:
        if not widths:
            return 0.0
        arr = np.asarray(widths, dtype=np.float32)
        q = float(np.clip(self.side_width_quantile, 0.5, 0.99))
        return float(np.quantile(arr, q))

    def _compute_repaired_side_asymmetry_signature(self, axis_angle: float, cx: float, cy: float, repaired_mask: np.ndarray) -> dict:
        ys, xs = np.nonzero(repaired_mask > 0)

        if len(xs) < 20:
            return {
                "valid": False,
                "area_pos": 0,
                "area_neg": 0,
                "robust_width_pos": 0.0,
                "robust_width_neg": 0.0,
                "area_term": 0.0,
                "width_term": 0.0,
                "score": 0.0,
                "suggested_angle360": self.angle_wrap_360(axis_angle),
            }

        theta = math.radians(axis_angle)
        ux, uy = math.cos(theta), math.sin(theta)
        nx, ny = -uy, ux

        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy

        u = dx * ux + dy * uy
        v = dx * nx + dy * ny

        area_pos = int(np.count_nonzero(v >= 0))
        area_neg = int(np.count_nonzero(v < 0))
        total_area = max(area_pos + area_neg, 1)

        umin = float(np.min(u))
        umax = float(np.max(u))
        if umax - umin < 1e-6:
            return {
                "valid": False,
                "area_pos": area_pos,
                "area_neg": area_neg,
                "robust_width_pos": 0.0,
                "robust_width_neg": 0.0,
                "area_term": 0.0,
                "width_term": 0.0,
                "score": 0.0,
                "suggested_angle360": self.angle_wrap_360(axis_angle),
            }

        n_bins = max(self.side_n_bins, 7)
        bins = np.linspace(umin, umax, n_bins + 1)

        widths_pos = []
        widths_neg = []

        for i in range(n_bins):
            m = (u >= bins[i]) & (u < bins[i + 1])
            if np.count_nonzero(m) < 3:
                continue

            vv = v[m]
            pos = vv[vv >= 0]
            neg = vv[vv < 0]

            if len(pos) > 0:
                widths_pos.append(float(np.max(pos)))
            if len(neg) > 0:
                widths_neg.append(float(-np.min(neg)))

        robust_width_pos = self._robust_side_width(widths_pos)
        robust_width_neg = self._robust_side_width(widths_neg)

        area_term = (area_neg - area_pos) / total_area
        denom_w = max(robust_width_pos + robust_width_neg, 1e-6)
        width_term = (robust_width_neg - robust_width_pos) / denom_w

        score = self.side_area_weight * area_term + self.side_width_weight * width_term

        if not self.side_flip_mapping:
            suggested = axis_angle if score >= 0 else axis_angle + 180.0
        else:
            suggested = axis_angle + 180.0 if score >= 0 else axis_angle

        return {
            "valid": True,
            "area_pos": area_pos,
            "area_neg": area_neg,
            "robust_width_pos": robust_width_pos,
            "robust_width_neg": robust_width_neg,
            "area_term": float(area_term),
            "width_term": float(width_term),
            "score": float(score),
            "suggested_angle360": self.angle_wrap_360(suggested),
        }

    def _compute_paste_side_signature(self, cell_det: dict, detections: List[dict], axis_angle: float, cx: float, cy: float) -> dict:
        expected_paste_name = self._expected_paste_name_for_cell(cell_det["name"])

        theta = math.radians(axis_angle)
        ux, uy = math.cos(theta), math.sin(theta)
        nx, ny = -uy, ux

        area_pos = 0.0
        area_neg = 0.0
        count_pos = 0
        count_neg = 0
        proj_pos = []
        proj_neg = []

        for d in detections:
            if d["name"] != expected_paste_name:
                continue

            overlap = self._overlap_ratio(d["mask"], cell_det["mask"])
            if overlap <= 0.0:
                continue

            inter = ((d["mask"] > 0) & (cell_det["mask"] > 0)).astype(np.uint8)
            inter_area = float(np.count_nonzero(inter))
            if inter_area <= 0:
                continue

            dx = float(d["cx"] - cx)
            dy = float(d["cy"] - cy)
            vproj = dx * nx + dy * ny

            if vproj >= 0:
                area_pos += inter_area
                count_pos += 1
                proj_pos.append(vproj)
            else:
                area_neg += inter_area
                count_neg += 1
                proj_neg.append(-vproj)

        total_area = max(area_pos + area_neg, 1.0)
        total_count = max(count_pos + count_neg, 1)

        area_term = (area_neg - area_pos) / total_area
        count_term = (count_neg - count_pos) / total_count

        mean_proj_pos = float(np.mean(proj_pos)) if proj_pos else 0.0
        mean_proj_neg = float(np.mean(proj_neg)) if proj_neg else 0.0
        denom_proj = max(mean_proj_pos + mean_proj_neg, 1e-6)
        proj_term = (mean_proj_neg - mean_proj_pos) / denom_proj

        score = (
            self.paste_count_weight * count_term
            + self.paste_area_weight * area_term
            + self.paste_proj_weight * proj_term
        )

        valid = (count_pos + count_neg) > 0

        if not self.paste_side_flip_mapping:
            suggested = axis_angle if score >= 0 else axis_angle + 180.0
        else:
            suggested = axis_angle + 180.0 if score >= 0 else axis_angle

        return {
            "valid": valid,
            "count_pos": int(count_pos),
            "count_neg": int(count_neg),
            "area_pos": float(area_pos),
            "area_neg": float(area_neg),
            "mean_proj_pos": float(mean_proj_pos),
            "mean_proj_neg": float(mean_proj_neg),
            "count_term": float(count_term),
            "area_term": float(area_term),
            "proj_term": float(proj_term),
            "score": float(score),
            "suggested_angle360": self.angle_wrap_360(suggested),
        }

    def _find_best_paste_for_cell(self, cell_det: dict, pastes: List[dict], axis_angle: float, cx: float, cy: float) -> Optional[dict]:
        theta_rad = math.radians(axis_angle)
        ux = math.cos(theta_rad)
        uy = math.sin(theta_rad)

        expected_paste_name = self._expected_paste_name_for_cell(cell_det["name"])
        best_paste = None
        best_key = None

        for p in pastes:
            if p["name"] != expected_paste_name:
                continue

            dist = float(np.hypot(cx - p["cx"], cy - p["cy"]))
            if dist > self.orientation_pair_max_dist:
                continue

            overlap = self._overlap_ratio(p["mask"], cell_det["mask"])
            if overlap <= 0.0:
                continue

            dx = float(p["cx"] - cx)
            dy = float(p["cy"] - cy)
            proj = dx * ux + dy * uy

            key = (float(overlap), -float(dist))
            if best_key is None or key > best_key:
                best_key = key
                best_paste = {
                    "det": p,
                    "dist": dist,
                    "overlap": overlap,
                    "proj": proj,
                }

        return best_paste

    def _find_nearest_track_for_memory(self, det: dict) -> Optional[dict]:
        best_tid = None
        best_dist = float("inf")

        for tid, tr in self.tracks.items():
            if tr["name"] != det["name"]:
                continue

            dist = math.hypot(
                float(det["cx"]) - float(tr["cx_smooth"]),
                float(det["cy"]) - float(tr["cy_smooth"])
            )
            if dist < best_dist and dist <= self.orientation_memory_max_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is None:
            return None

        tr = self.tracks[best_tid]
        out = dict(tr)
        out["_dist"] = best_dist
        return out

    def _choose_angle_with_memory(self, cand_angle: float, prev_angle: Optional[float], confidence: float) -> float:
        cand_angle = self.angle_wrap_360(cand_angle)
        if prev_angle is None:
            return cand_angle

        prev_angle = self.angle_wrap_360(prev_angle)
        diff = self.angle_diff_deg_360(cand_angle, prev_angle)

        if diff >= self.orientation_flip_guard_deg and confidence < self.orientation_keep_prev_margin:
            return prev_angle

        return cand_angle

    def _assign_360_orientation(self, detections: List[dict]) -> List[dict]:
        """
        Assign 0~360 orientation for PV cells.

        Improvement:
            1. Keep your original cue voting logic:
            repaired_side / paste_side / end_shape / paste_weak.
            2. Add low-confidence guard:
            if score_same and score_flip are too close, do not randomly flip.
            3. Add same-row consistency:
            if two PV cells in the same row have the same axis direction,
            but one is 270 and one is 90, force the low-confidence one
            to follow the high-confidence one.

        This fixes the case:
            slot0 = 270 correct
            slot1 = 90 wrong
            but both PV cells physically have the same direction.
        """

        if not detections:
            return detections

        pastes = [
            d for d in detections
            if d["name"] in ("Paste area", "Paste area down")
        ]

        # Store PV candidates for a second-pass row consistency check.
        pv_for_consistency = []

        # These are intentionally conservative.
        min_orientation_confidence = float(
            self.params.get("min_orientation_confidence", 0.08)
        )
        row_consistency_enable = bool(
            self.params.get("place_row_orientation_consistency_enable", True)
        )
        row_consistency_y_tol_px = float(
            self.params.get("place_row_orientation_consistency_y_tol_px", 80.0)
        )
        row_consistency_axis_tol_deg = float(
            self.params.get("place_row_orientation_consistency_axis_tol_deg", 25.0)
        )
        row_consistency_flip_min_diff_deg = float(
            self.params.get("place_row_orientation_consistency_flip_min_diff_deg", 120.0)
        )

        for d in detections:
            if d["name"] not in ("PV cell up", "PV cell down"):
                continue

            masks = self._make_orientation_masks(d, detections)
            raw_cell_mask = masks["raw_cell_mask"]
            minus_paste_mask = masks["minus_paste_mask"]
            repaired_mask = masks["repaired_mask"]

            axis_info = self._compute_axis_from_repaired_mask(repaired_mask)

            if axis_info["valid"]:
                axis_angle = float(axis_info["axis_angle"])
                cx_ori = float(axis_info["cx"])
                cy_ori = float(axis_info["cy"])
                aspect_ratio_ori = float(axis_info["aspect_ratio"])
            else:
                axis_angle = float(d["axis_angle"])
                cx_ori = float(d["cx"])
                cy_ori = float(d["cy"])
                aspect_ratio_ori = float(d.get("aspect_ratio", 0.0))

            d["axis_angle"] = axis_angle
            d["cx_orientation"] = cx_ori
            d["cy_orientation"] = cy_ori
            d["aspect_ratio_orientation"] = aspect_ratio_ori

            # Keep your original behavior:
            # use orientation center as cell center.
            d["cx"] = cx_ori
            d["cy"] = cy_ori

            side_sig = self._compute_repaired_side_asymmetry_signature(
                axis_angle=axis_angle,
                cx=cx_ori,
                cy=cy_ori,
                repaired_mask=repaired_mask,
            )

            side_valid = (
                bool(side_sig["valid"])
                and abs(float(side_sig["score"])) >= self.side_score_min
            )

            paste_side_sig = self._compute_paste_side_signature(
                cell_det=d,
                detections=detections,
                axis_angle=axis_angle,
                cx=cx_ori,
                cy=cy_ori,
            )

            paste_side_valid = (
                self.use_paste_side_cue
                and bool(paste_side_sig["valid"])
                and abs(float(paste_side_sig["score"])) >= self.paste_side_score_min
            )

            end_sig = self._compute_end_shape_signature(
                axis_angle=axis_angle,
                cx=cx_ori,
                cy=cy_ori,
                ref_mask=repaired_mask,
            )

            end_valid = (
                bool(end_sig["valid"])
                and float(end_sig["ratio_diff"]) >= self.end_shape_min_ratio_diff
            )

            best_paste = None
            if self.use_paste_as_weak_cue and pastes:
                best_paste = self._find_best_paste_for_cell(
                    cell_det=d,
                    pastes=pastes,
                    axis_angle=axis_angle,
                    cx=cx_ori,
                    cy=cy_ori,
                )

            votes = []

            if side_valid:
                votes.append({
                    "angle": float(side_sig["suggested_angle360"]),
                    "weight": self.side_vote_weight * abs(float(side_sig["score"])),
                    "source": "repaired_side",
                })

            if paste_side_valid:
                votes.append({
                    "angle": float(paste_side_sig["suggested_angle360"]),
                    "weight": self.paste_side_vote_weight * abs(float(paste_side_sig["score"])),
                    "source": "paste_side",
                })

            if end_valid:
                votes.append({
                    "angle": float(end_sig["suggested_angle360"]),
                    "weight": self.end_shape_vote_weight * float(end_sig["ratio_diff"]),
                    "source": "end_shape",
                })

            if best_paste is not None:
                proj = float(best_paste["proj"])
                overlap = float(best_paste["overlap"])

                if abs(proj) >= self.orientation_min_proj_abs:
                    paste_angle = self.angle_wrap_360(
                        axis_angle if proj >= 0 else axis_angle + 180.0
                    )

                    votes.append({
                        "angle": paste_angle,
                        "weight": self.paste_vote_weight * max(overlap, 1e-6),
                        "source": "paste_weak",
                    })

            angle_same = self.angle_wrap_360(axis_angle)
            angle_flip = self.angle_wrap_360(axis_angle + 180.0)

            score_same = 0.0
            score_flip = 0.0
            used_sources = []

            if not votes:
                chosen_angle = angle_same
                chosen_source = "axis_only"
                confidence = 0.0
            else:
                for v in votes:
                    a = self.angle_wrap_360(v["angle"])
                    w = float(v["weight"])

                    if self.angle_diff_deg_360(a, angle_same) <= 45.0:
                        score_same += w
                    else:
                        score_flip += w

                    used_sources.append(v["source"])

                confidence = abs(score_same - score_flip)
                chosen_source = "+".join(sorted(set(used_sources)))

                # ----------------------------------------------------
                # Important improvement:
                # If same/flip score is too close, do NOT randomly choose.
                # Prefer memory if available; otherwise keep the stronger one.
                # ----------------------------------------------------
                if confidence < min_orientation_confidence:
                    chosen_angle = angle_same if score_same >= score_flip else angle_flip
                    chosen_source = chosen_source + "+low_conf_vote"

                else:
                    chosen_angle = angle_same if score_same >= score_flip else angle_flip

            prev_angle = None
            chosen_before_memory = self.angle_wrap_360(chosen_angle)
            final_angle = self.angle_wrap_360(chosen_angle)

            d["angle"] = final_angle
            d["angle_source"] = chosen_source

            d["raw_cell_area"] = int(np.count_nonzero(raw_cell_mask))
            d["minus_paste_area"] = int(np.count_nonzero(minus_paste_mask))
            d["repaired_area"] = int(np.count_nonzero(repaired_mask))

            # Extra debug fields.
            d["orientation_axis_angle"] = float(axis_angle)
            d["orientation_angle_same"] = float(angle_same)
            d["orientation_angle_flip"] = float(angle_flip)
            d["orientation_score_same"] = float(score_same)
            d["orientation_score_flip"] = float(score_flip)
            d["orientation_confidence"] = float(confidence)
            d["orientation_chosen_before_memory"] = float(chosen_before_memory)
            d["orientation_prev_angle"] = None if prev_angle is None else float(prev_angle)
            d["orientation_final_angle"] = float(final_angle)

            if self.debug:
                print(
                    f"[ORI DEBUG] name={d['name']} "
                    f"cx={d['cx']:.1f}, cy={d['cy']:.1f}, "
                    f"axis={axis_angle:.1f}, "
                    f"same={angle_same:.1f}, flip={angle_flip:.1f}, "
                    f"score_same={score_same:.4f}, "
                    f"score_flip={score_flip:.4f}, "
                    f"confidence={confidence:.4f}, "
                    f"prev={prev_angle}, "
                    f"before_mem={chosen_before_memory:.1f}, "
                    f"final={final_angle:.1f}, "
                    f"source={chosen_source}"
                )

            pv_for_consistency.append(d)

        # ------------------------------------------------------------
        # Second pass:
        # same-row orientation consistency.
        #
        # If slot0 and slot1 are on the same row and have the same axis,
        # but one is 270 and one is 90, use the higher-confidence result
        # as anchor and flip the weaker one.
        # ------------------------------------------------------------
        if row_consistency_enable and len(pv_for_consistency) >= 2:
            # Sort top-to-bottom, then left-to-right.
            pv_sorted = sorted(
                pv_for_consistency,
                key=lambda x: (
                    float(x.get("cy", 0.0)),
                    float(x.get("cx", 0.0)),
                )
            )

            groups = []

            for d in pv_sorted:
                placed = False
                cy = float(d.get("cy", 0.0))

                for g in groups:
                    g_cy = float(np.mean([float(x.get("cy", 0.0)) for x in g]))

                    if abs(cy - g_cy) <= row_consistency_y_tol_px:
                        g.append(d)
                        placed = True
                        break

                if not placed:
                    groups.append([d])

            for g in groups:
                if len(g) < 2:
                    continue

                # Choose the most reliable PV as anchor.
                anchor = max(
                    g,
                    key=lambda x: (
                        float(x.get("orientation_confidence", 0.0)),
                        float(x.get("score", 0.0)),
                        float(x.get("area", 0.0)),
                    )
                )

                anchor_angle = self.angle_wrap_360(float(anchor.get("angle", 0.0)))
                anchor_axis = self.angle_wrap_180(float(anchor.get("axis_angle", 0.0)))

                for d in g:
                    if d is anchor:
                        continue

                    cur_angle = self.angle_wrap_360(float(d.get("angle", 0.0)))
                    cur_axis = self.angle_wrap_180(float(d.get("axis_angle", 0.0)))

                    axis_diff = abs((cur_axis - anchor_axis + 90.0) % 180.0 - 90.0)
                    angle_diff = self.angle_diff_deg_360(cur_angle, anchor_angle)

                    if axis_diff <= row_consistency_axis_tol_deg and angle_diff >= row_consistency_flip_min_diff_deg:
                        old_angle = cur_angle

                        d["angle_raw_before_row_consistency"] = float(old_angle)
                        d["angle"] = float(anchor_angle)
                        d["orientation_final_angle"] = float(anchor_angle)
                        d["angle_source"] = str(d.get("angle_source", "")) + "+row_consistency"

                        if self.debug:
                            print(
                                f"[ORI ROW FIX] "
                                f"name={d['name']} "
                                f"cx={d['cx']:.1f}, cy={d['cy']:.1f}, "
                                f"old_angle={old_angle:.1f}, "
                                f"new_angle={anchor_angle:.1f}, "
                                f"anchor=({anchor.get('cx', 0.0):.1f},{anchor.get('cy', 0.0):.1f}), "
                                f"axis_diff={axis_diff:.1f}, "
                                f"angle_diff={angle_diff:.1f}"
                            )

        return detections

    def create_track(self, det: dict):
        tid = self.next_track_id
        self.next_track_id += 1

        self.tracks[tid] = {
            "id": tid,
            "name": det["name"],
            "cx": float(det["cx"]),
            "cy": float(det["cy"]),
            "cx_smooth": float(det["cx"]),
            "cy_smooth": float(det["cy"]),
            "angle": float(det.get("angle", 0.0)),
            "angle_smooth": float(det.get("angle", 0.0)),
            "score": float(det.get("score", 0.0)),
            "hits": 1,
            "age": 0,
            "stable_count": 1,
            "last_seen": time.time(),
        }
        return tid

    def update_track(self, tid: int, det: dict):
        tr = self.tracks[tid]

        prev_cx = float(tr["cx_smooth"])
        prev_cy = float(tr["cy_smooth"])
        prev_angle = float(tr["angle_smooth"])

        det_cx = float(det["cx"])
        det_cy = float(det["cy"])
        det_angle = float(det.get("angle", 0.0))

        tr["cx"] = det_cx
        tr["cy"] = det_cy
        tr["angle"] = det_angle
        tr["score"] = float(det.get("score", 0.0))
        tr["name"] = det["name"]

        tr["cx_smooth"] = (1.0 - self.ema_alpha) * prev_cx + self.ema_alpha * det_cx
        tr["cy_smooth"] = (1.0 - self.ema_alpha) * prev_cy + self.ema_alpha * det_cy
        tr["angle_smooth"] = self.smooth_angle_360(prev_angle, det_angle)

        pos_jitter = math.hypot(det_cx - prev_cx, det_cy - prev_cy)
        ang_jitter = self.angle_diff_deg_360(det_angle, prev_angle)

        if pos_jitter <= self.position_jitter_tol and ang_jitter <= self.angle_jitter_tol:
            tr["stable_count"] += 1
        else:
            tr["stable_count"] = 1

        tr["hits"] += 1
        tr["age"] = 0
        tr["last_seen"] = time.time()
        return tr

    def match_and_update_tracks(self, detections: List[dict]):
        for det in detections:
            det["_track_id"] = None
            det["_track_stable"] = False

        assigned_tracks = set()
        assigned_dets = set()
        pair_candidates = []

        for di, det in enumerate(detections):
            det_name = det["name"]
            det_cx = float(det["cx"])
            det_cy = float(det["cy"])
            det_angle = float(det.get("angle", 0.0))

            for tid, tr in self.tracks.items():
                if tr["name"] != det_name:
                    continue

                dist = math.hypot(
                    det_cx - float(tr["cx_smooth"]),
                    det_cy - float(tr["cy_smooth"])
                )
                if dist > self.pixel_gate:
                    continue

                ang_diff = self.angle_diff_deg_360(
                    det_angle,
                    float(tr["angle_smooth"])
                )

                cost = dist + 0.15 * ang_diff

                pair_candidates.append((
                    float(cost),
                    float(dist),
                    float(ang_diff),
                    -int(tr["hits"]),
                    di,
                    tid,
                ))

        pair_candidates.sort()

        for _, _, _, _, di, tid in pair_candidates:
            if di in assigned_dets or tid in assigned_tracks:
                continue

            det = detections[di]
            tr = self.update_track(tid, det)

            assigned_dets.add(di)
            assigned_tracks.add(tid)

            det["_track_id"] = int(tid)
            det["_track_stable"] = bool(
                tr["hits"] >= self.min_hits and
                tr["stable_count"] >= self.stable_required
            )

        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue

            tid = self.create_track(det)
            tr = self.tracks[tid]

            assigned_dets.add(di)
            assigned_tracks.add(tid)

            det["_track_id"] = int(tid)
            det["_track_stable"] = bool(
                tr["hits"] >= self.min_hits and
                tr["stable_count"] >= self.stable_required
            )

        to_delete = []
        for tid, tr in self.tracks.items():
            if tid not in assigned_tracks:
                tr["age"] += 1
                if tr["age"] > self.max_age:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

    def prune_tracks_near_point(self, cx: float, cy: float, radius_px: float, names: Optional[List[str]] = None):
        to_delete = []
        for tid, tr in self.tracks.items():
            if names is not None and tr["name"] not in names:
                continue
            dist = math.hypot(tr["cx_smooth"] - cx, tr["cy_smooth"] - cy)
            if dist <= radius_px:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

    def clear_all_tracks(self):
        self.tracks.clear()
        self.next_track_id = 1



    def process_frame(self, frame: np.ndarray, roi_mode: str = "default"):
        self.frame_count += 1

        full_frame, infer_frame, (ox, oy), roi_box = self.preprocess_roi(
            frame, roi_mode=roi_mode
        )

        vis = full_frame.copy()
        roi_h, roi_w = infer_frame.shape[:2]
        full_h, full_w = full_frame.shape[:2]

        center_x = full_w // 2
        center_y = full_h // 2

        x = self._prepare_input_tensor(infer_frame)
        model_inputs = ({"image": x},)

        with torch.no_grad():
            outputs = self.model(model_inputs)

        boxes, classes, masks, scores = self._validate_outputs(
            outputs, image_h=roi_h, image_w=roi_w
        )

        boxes = boxes.cpu().numpy()
        classes = classes.cpu().numpy()
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

        detections = []

        # ===============================
        # DETECTIONS
        # ===============================


        n = min(len(scores), len(masks), len(boxes), len(classes))

        for i in range(n):

            if i >= len(masks) or masks[i] is None:
                continue

            if i >= len(boxes):
                continue

            score = float(scores[i])
            if score < self.score_thresh:
                continue

            class_id = int(classes[i])
            name = self.class_names[class_id]

            if name not in self.target_class_name_set:
                continue


            full_mask_roi = self._mask_to_full_image(
                mask_raw=masks[i],
                box=boxes[i],
                image_h=roi_h,
                image_w=roi_w,
                thresh=float(self.mask_thresh),
            )

            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            full_mask[oy:oy + roi_h, ox:ox + roi_w] = full_mask_roi

            geom = self.mask_to_geometry(full_mask, name=name)
            if geom is None:
                continue

            detections.append({
                "name": name,
                "score": score,
                "cx": float(geom["cx"]),
                "cy": float(geom["cy"]),
                "rect_pts": geom["rect_pts"],
                "mask": full_mask,
            })

        # ===============================
        # TRACKING
        # ===============================
        self.match_and_update_tracks(detections)


        # ===============================
        # ✅ FILTER ALREADY PASSED CELLS
        # ===============================
        filtered_detections = []

        for det in detections:
            cx = det["cx"]
            cy = det["cy"]

            already_passed = False
            for p in self.passed_cells:
                dist = math.hypot(cx - p["cx"], cy - p["cy"])
                if dist < self.passed_match_radius:
                    already_passed = True
                    break

            if not already_passed:
                filtered_detections.append(det)

        detections = filtered_detections


        # ===============================
        # ✅ FILTER PASSED CELLS (NO REUSE)
        # ===============================
        filtered_detections = []

        for det in detections:
            cx = det["cx"]
            cy = det["cy"]

            already_passed = False

            for p in self.passed_cells:
                dist = math.hypot(cx - p["cx"], cy - p["cy"])
                if dist < self.passed_match_radius:
                    already_passed = True
                    break

            if not already_passed:
                filtered_detections.append(det)

        detections = filtered_detections


        # ===============================
        # ✅ FIXED: TARGET SELECTION
        # ===============================
        self.current_target_id = None
        best_det = None
        best_idx = None

        if self.centering_active and detections:



            candidates = [
                (i, d) for i, d in enumerate(detections)
                if d.get("_track_id") is not None and d["cy"] >= center_y
            ]




            if candidates:
                below = [(i, d) for i, d in candidates if d["cy"] >= center_y]

                if below:
                    best_idx, best_det = min(
                        below, key=lambda x: abs(x[1]["cy"] - center_y)
                    )
                else:
                    best_idx, best_det = min(
                        candidates, key=lambda x: abs(x[1]["cy"] - center_y)
                    )

                self.current_target_id = best_det["_track_id"]

        # ===============================
        # ✅ FIXED: CENTERING + NUMBERING
        # ===============================
        offset_dx = 0.0
        offset_dy = 0.0

        if best_det is not None:
            cx = float(best_det["cx"])
            cy = float(best_det["cy"])

            offset_dx = cx - center_x
            offset_dy = cy - center_y
            dist = math.hypot(offset_dx, offset_dy)

            if dist < self.centered_threshold_px:
                self._centered_count += 1
            else:
                self._centered_count = 0

            if self._centered_count >= self.centered_confirm_frames:
  

                print("CENTER TRIGGERED")

                # ✅ trigger ONLY ONCE
                if self.centered_message_timer == 0:
                    self.centered_message_timer = self.centered_message_duration

                self.cell_centered = True
                self._centered_count = 0
                self.current_target_id = None



                # ✅ store passed cell
                self.passed_cells.append({
                    "cx": cx,
                    "cy": cy
                })

                if len(self.passed_cells) > 10:
                    self.passed_cells = self.passed_cells[-10:]




                self.cell_centered = True
                self._centered_count = 0
                self.current_target_id = None
            else:
                self.cell_centered = False

        else:
            self._centered_count = 0
            self.cell_centered = False

        # ===============================
        # DRAW
        # ===============================
        for det in detections:

            if det["cy"] < center_y:
                continue

            cx, cy = int(det["cx"]), int(det["cy"])
            rect_pts = np.asarray(det["rect_pts"], dtype=np.int32)

            if det.get("_track_id") == self.current_target_id:
                cv2.drawContours(vis, [rect_pts], -1, (0, 0, 255), 4)
                cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            else:
                cv2.drawContours(vis, [rect_pts], -1, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)




            # ===============================
            # ✅ DRAW PASSED CELLS (NO NUMBER)
            # ===============================
            for p in self.passed_cells:
                cx, cy = int(p["cx"]), int(p["cy"])

                # draw small faded circle instead of rectangle
                cv2.circle(vis, (cx, cy), 4, (80, 80, 80), 1)



        # ===============================
        # CROSSHAIR
        # ===============================

        cv2.line(vis, (0, center_y), (full_w, center_y), (255, 255, 0), 1)
        cv2.line(vis, (center_x, 0), (center_x, full_h), (255, 255, 0), 1)

        # ✅ center point marker
        cv2.circle(vis, (center_x, center_y), 6, (0, 0, 255), -1)

        # ✅ dynamic threshold circle (important for centering)
        cv2.circle(vis, (center_x, center_y), int(self.centered_threshold_px), (0, 255, 255), 1)


        centered_progress = self._centered_count / max(self.centered_confirm_frames, 1)



        return vis, {
            "detections": detections,
            "offset_dx": offset_dx,
            "offset_dy": offset_dy,
            "cell_centered": self.cell_centered,
            "centered_progress": centered_progress,   
            "centering_active": self.centering_active
        }


       

        # ===============================
        # ✅ DISPLAY "CENTERED" MESSAGE
        # ===============================
        if self.centered_message_timer > 0:

            text = "CENTERED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness = 3

            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

            x = int((full_w - tw) / 2)
            y = int(full_h * 0.2)

            # ✅ draw background box
            cv2.rectangle(vis,
                        (x - 15, y - th - 15),
                        (x + tw + 15, y + 10),
                        (0, 0, 0), -1)

            # ✅ draw text
            cv2.putText(vis, text, (x, y),
                        font, scale, (0, 255, 0), thickness)

            # ✅ decrease timer LAST
            self.centered_message_timer -= 1


            # ===============================
            # ✅ DRAW CENTERED MESSAGE (FINAL LAYER)
            # ===============================
            if self.centered_message_timer > 0:

                text = "CENTERED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.5
                thickness = 3

                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

                x = int((full_w - tw) / 2)
                y = int(full_h * 0.2)

                # draw background
                cv2.rectangle(vis,
                            (x - 20, y - th - 20),
                            (x + tw + 20, y + 10),
                            (0, 0, 0), -1)

                # draw text (bright green)
                cv2.putText(vis, text, (x, y),
                            font, scale, (0, 255, 0), thickness)

                # decrease AFTER drawing
                self.centered_message_timer -= 1


    def _filter_paste_by_shape(self, detections: List[dict]) -> List[dict]:
        out = []
        for d in detections:
            if d["name"] in ("Paste area", "Paste area down"):
                area = float(d.get("area", 0.0))
                ar = float(d.get("aspect_ratio", 0.0))

                if area < self.paste_area_min_area or area > self.paste_area_max_area:
                    continue

                if not (self.paste_area_aspect_min <= ar <= self.paste_area_aspect_max):
                    continue

            out.append(d)
        return out



def main():
    script_dir = Path(__file__).resolve().parent
    cfg = load_config(script_dir / "2.json")

    webcam_index = int(cfg.get("webcam_index", 0))
    camera_width = int(cfg.get("camera_width", 640))
    camera_height = int(cfg.get("camera_height", 480))

    processor = TorchScriptCameraProcessor(
        model_path=cfg["weight_path"],
        params=cfg["camera2_params"]
    )

    cap = cv2.VideoCapture(webcam_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {webcam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    win_name = "PV cell up/down video stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 900)

    prev_t = cv2.getTickCount()
    print("[INFO] Controls: p = pause/resume centering, d = pause/resume detection, 1 = left, 2 = center, 3 = right, q = quit")

    def _set_centering_state(active: bool, reset_counter: bool = True) -> bool:
        if hasattr(processor, "set_centering_active"):
            processor.set_centering_active(bool(active), reset_counter=reset_counter)
            return bool(getattr(processor, "centering_active", bool(active)))
        processor.centering_active = bool(active)
        if reset_counter and hasattr(processor, "_centered_count"):
            processor._centered_count = 0
        if hasattr(processor, "cell_centered"):
            processor.cell_centered = False
        return bool(getattr(processor, "centering_active", bool(active)))

    ui_centering_active = _set_centering_state(True, reset_counter=True)
    ui_detection_active = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from webcam.")
                continue

            target_mode = str(getattr(processor, "target_mode", "center") or "center")
            detections_count = 0
            offset_dx = 0.0
            offset_dy = 0.0
            centered_progress = 0.0
            if ui_detection_active:
                vis, det_out = processor.process_frame(frame)
                if isinstance(det_out, dict):
                    ui_centering_active = bool(det_out.get("centering_active", ui_centering_active))
                    target_mode = str(det_out.get("target_mode", target_mode) or target_mode)
                    dets = det_out.get("detections", []) or []
                    detections_count = len(dets) if isinstance(dets, list) else 0
                    offset_dx = float(det_out.get("offset_dx", 0.0))
                    offset_dy = float(det_out.get("offset_dy", 0.0))
                    centered_progress = float(det_out.get("centered_progress", 0.0))
                else:
                    ui_centering_active = bool(getattr(processor, "centering_active", ui_centering_active))
            else:
                vis = frame.copy()
                ui_centering_active = bool(getattr(processor, "centering_active", ui_centering_active))

            now_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(now_t - prev_t, 1)
            prev_t = now_t

            # Unified top-left HUD box.
            panel_x = 12
            panel_y = 12
            panel_w = min(320, max(250, vis.shape[1] - panel_x - 12))
            panel_h = 147
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (110, 110, 110), 1)

            center_state = "ACTIVE" if ui_centering_active else "PAUSED"
            detect_state = "ACTIVE" if ui_detection_active else "PAUSED"
            lines = [
                (f"FPS: {fps:.1f}", (0, 255, 255)),
                (f"Offset: dx={int(round(offset_dx))} dy={int(round(offset_dy))}", (0, 255, 255)),
                (f"Detections: {int(detections_count)}", (255, 255, 0)),
                (f"Centering: {center_state} ({centered_progress:.0%})", (0, 255, 0) if ui_centering_active else (0, 200, 255)),
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

            cv2.imshow(win_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")):
                ui_detection_active = not ui_detection_active
                ui_centering_active = _set_centering_state(not ui_centering_active, reset_counter=True)
                print(
                    f"[CONTROL] Pause toggle -> centering={'ACTIVE' if ui_centering_active else 'PAUSED'}, "
                    f"detection={'ACTIVE' if ui_detection_active else 'PAUSED'}"
                )
            elif key in (ord("d"), ord("D")):
                ui_detection_active = not ui_detection_active
                print(f"[DETECT] Detection {'resumed' if ui_detection_active else 'paused'} by user.")
            elif key == ord("1"):
                processor.target_mode = "left"
                print("[CENTER] Target mode set to LEFT.")
            elif key == ord("2"):
                processor.target_mode = "center"
                print("[CENTER] Target mode set to CENTER.")
            elif key == ord("3"):
                processor.target_mode = "right"
                print("[CENTER] Target mode set to RIGHT.")
            elif key in (ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()