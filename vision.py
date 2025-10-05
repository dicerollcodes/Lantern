"""
Real-time YOLO + MiDaS from webcam 
- YOLOv8n detection + MiDaS (midas_v21_small) depth.
- Overlays: bounding boxes, labels, per-box relative depth, global depth heatmap.
- Controls: q=quit, d=toggle depth, m=cycle overlay mode, r=record, b=toggle boxes, i=toggle HUD.

Author: VisionAssist MVP bootstrap
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ---- YOLO (Ultralytics) ----
from ultralytics import YOLO

# ---- Torch performance knobs ----
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def init_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------- MiDaS ----------
def init_midas(device: torch.device, model_type: str = "DPT_Large"):
    """
    Load MiDaS via torch.hub.
    model_type: "DPT_Large" (high quality), "DPT_Hybrid" (balanced), "midas_v21_small" (fast)
    """
    # The MiDaS repository exposes callables with names like
    # "MiDaS_small", "MiDaS", "DPT_Hybrid", "DPT_Large", etc.
    # Older scripts often use names like "midas_v21_small" — map common
    # CLI values to the actual hubcallable names here.
    mt = model_type.strip()
    mt_l = mt.lower()
    if mt_l in ("midas_v21_small", "midas_small", "small", "midas_v21small"):
        hub_name = "MiDaS_small"
    elif mt_l in ("midas_v21", "midas_v21_384", "midas", "midas_v21_384"):
        hub_name = "MiDaS"
    elif mt.upper().startswith("DPT_"):
        # DPT_* callables are exported with the same casing
        hub_name = model_type
    else:
        # fallback: try the provided string (may raise the original error)
        hub_name = model_type

    midas = torch.hub.load("intel-isl/MiDaS", hub_name)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "small" in model_type or "v21" in model_type:
        transform = transforms.small_transform
    else:
        transform = transforms.dpt_transform

    return midas, transform


def run_midas(midas, transform, frame_bgr, device):
    """Return depth map as float32 numpy array (higher = closer, relative)."""
    # MiDaS expects RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.inference_mode():
        pred = midas(input_batch)
        # (N, H, W) -> upsample to original frame size
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth = pred[0].detach().cpu().numpy().astype(np.float32)  # HxW
    return depth


def depth_to_colormap(depth: np.ndarray, clip: float = 0.0) -> np.ndarray:
    """
    Normalize inverse depth to [0, 255] and apply a colormap.
    clip>0 will clip top percentile to improve contrast.
    """
    d = depth.copy()
    if clip > 0:
        hi = np.percentile(d, 100 - clip)
        d = np.clip(d, 0, hi)
    d -= d.min()
    rng = d.max() if d.max() > 1e-6 else 1.0
    d = (d / rng * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    return cm


# ---------- YOLO ----------
def init_yolo(weights: str, device: torch.device, half: bool = True):
    """Initializes the YOLO model and moves it to the specified device."""
    model = YOLO(weights)
    model.to(device)
    return model



def run_yolo(model, frame_bgr, imgsz: int, conf: float, device: torch.device):
    """
    Returns: list of dict with x1,y1,x2,y2,conf,cls_id,cls_name
    """
    # The 'half' argument should be passed from the main loop's args
    # For this function, let's assume it's a parameter.
    # We will adjust the call in main()
    # For now, let's just make the change here.
    
    # Ultralytics accepts device as 'cpu' or 'cuda:0' etc. Build device string from torch.device
    if isinstance(device, torch.device):
        if device.type == "cuda":
            device_str = f"cuda:{device.index if getattr(device, 'index', None) is not None else 0}"
        else:
            device_str = "cpu"
    else:
        device_str = str(device)

    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=device_str,
    )[0]


    boxes = []
    if results.boxes is None:
        return boxes

    xyxy = results.boxes.xyxy.detach().cpu().numpy()
    confs = results.boxes.conf.detach().cpu().numpy()
    clss = results.boxes.cls.detach().cpu().numpy().astype(int)
    names = results.names if hasattr(results, "names") else model.model.names

    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        boxes.append(
            dict(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                conf=float(c),
                cls_id=int(k),
                cls_name=str(names.get(int(k), str(int(k)))),
            )
        )
    return boxes


# ---------- Drawing / HUD ----------
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1, bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_boxes(frame, boxes, depth=None, depth_near: float = 0.2, depth_far: float = 5.0):
    H, W = frame.shape[:2]
    # Compute per-frame quantiles for relative distance words
    depth_q = None
    if depth is not None:
        flat = depth.reshape(-1)
        depth_q = (np.percentile(flat, 35), np.percentile(flat, 65))  # near/medium/far splits

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cls_name, conf = b["cls_name"], b["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        
        # Draw midpoint of bounding box
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.circle(frame, (mid_x, mid_y), 4, (0, 255, 255), -1)  # Yellow filled circle
        cv2.circle(frame, (mid_x, mid_y), 6, (0, 0, 0), 2)       # Black outline
        
        depth_word = ""
        if depth is not None:
            d_roi = depth[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if d_roi.size > 0:
                med = float(np.median(d_roi))
                # higher = closer for MiDaS inverse depth
                # Compute per-frame robust min/max to normalize (avoid outliers)
                gmin = float(np.percentile(depth.reshape(-1), 1))
                gmax = float(np.percentile(depth.reshape(-1), 99))
                if gmax - gmin > 1e-6:
                    depth_norm = (med - gmin) / (gmax - gmin)
                    # depth_norm: 1.0 = closest, 0.0 = farthest
                else:
                    depth_norm = 0.5
                # Map normalized inverse depth to approximate meters using assumed near/far
                # We'll attach depth_m later via caller-provided mapping if desired.
                b["rel_depth_median"] = med
                b["depth_norm"] = depth_norm
                # approximate mapping to meters (assumes inverse depth normalized)
                try:
                    meters = float(depth_far) - depth_norm * (float(depth_far) - float(depth_near))
                except Exception:
                    meters = float(depth_near)
                b["depth_m"] = meters

        label = f"{cls_name} {conf:.2f}"
        # If we computed numeric depth, append it as centimeters
        if "depth_m" in b:
            cm = int(round(b["depth_m"] * 100.0))
            label += f" • {cm}cm"
        elif depth_word:
            label += f" • {depth_word}"
        put_text(frame, label, (x1 + 2, max(18, y1 - 6)))

    return frame


def blend_depth(frame_bgr, depth_cm, alpha=0.55):
    depth_resized = cv2.resize(depth_cm, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, depth_resized, alpha, 0)


# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser(description="Real-time YOLO + MiDaS from webcam (no WebSockets)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--yolo", type=str, default="yolov8m.pt", help="YOLO weights (e.g., yolov8n.pt, yolov8m.pt, or yolov10n.pt)")
    ap.add_argument("--midas", type=str, default="DPT_Large", help="MiDaS model: DPT_Large | DPT_Hybrid | MiDaS_small")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for both models")
    ap.add_argument("--no-half", action="store_true", help="Disable half precision for YOLO (GPU)")
    ap.add_argument("--depth-every", type=int, default=1, help="Run depth every N frames (>=1).")
    ap.add_argument("--live-depth", action="store_true", help="Run depth every frame (may be slower) and update overlay live")
    ap.add_argument("--depth-near", type=float, default=0.2, help="Assumed near distance in meters for numeric depth mapping (default 0.2m)")
    ap.add_argument("--depth-far", type=float, default=5.0, help="Assumed far distance in meters for numeric depth mapping (default 5.0m)")
    ap.add_argument("--overlay", type=str, default="blend", choices=["blend", "side"], help="Depth overlay mode")
    ap.add_argument("--alpha", type=float, default=0.55, help="Depth blend alpha")
    ap.add_argument("--save", type=str, default="", help="Optional path to save output video (e.g., out.mp4)")
    args = ap.parse_args()

    device = init_device(force_cpu=args.cpu)
    print(f"[Init] Device: {device}")
    print("[Init] Loading YOLO:", args.yolo)
    yolo = init_yolo(args.yolo, device, half=not args.no_half)
    # Warm-up YOLO predictor on the chosen device once to avoid heavy setup during the first frame
    try:
        device_arg = f"cuda:{device.index if hasattr(device, 'index') and device.index is not None else 0}" if device.type == "cuda" else "cpu"
        print(f"[Init] Warming up YOLO on device: {device_arg}")
        _img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        _t0 = time.time()
        # run a single prediction to ensure model is set up and fused on the GPU
        yolo.predict(source=_img, imgsz=args.imgsz, conf=args.conf, verbose=False, device=device_arg)
        print(f"[Init] YOLO warmup took {time.time() - _t0:.3f}s")
    except Exception as e:
        print("[Warn] YOLO warmup failed:", e)
    print("[Init] Loading MiDaS:", args.midas)
    midas, midas_transform = init_midas(device, args.midas)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # helps on many webcams

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (int(args.width), int(args.height)))
        if not writer.isOpened():
            print("[Warn] Could not open writer; recording disabled.")
            writer = None

    show_boxes = True
    show_hud = True
    show_depth = True
    overlay_mode = args.overlay  # 'blend' or 'side'
    depth_stride = max(1, args.depth_every)

    # FPS accounting
    t_prev = time.time()
    smoothed_fps = 0.0
    yolo_times = deque(maxlen=30)
    depth_times = deque(maxlen=30)
    frame_idx = 0
    last_depth = None
    last_depth_cm = None

    window_name = "VisionAssist — YOLO + MiDaS (q=quit, d=depth, m=mode, r=rec, b=boxes, i=HUD)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Warn] Empty frame")
                continue

            frame_idx += 1
            H, W = frame.shape[:2]

            # YOLO
            t0 = time.time()
            boxes = run_yolo(yolo, frame, imgsz=args.imgsz, conf=args.conf, device=device)
            yolo_times.append(time.time() - t0)

            # MiDaS (every N frames)
            # Decide whether to run MiDaS this frame
            if args.live_depth:
                do_depth = True
            else:
                do_depth = (frame_idx % depth_stride == 1) or (last_depth is None)

            if do_depth:
                t0 = time.time()
                depth = run_midas(midas, midas_transform, frame, device)
                depth_times.append(time.time() - t0)
                last_depth = depth
                last_depth_cm = depth_to_colormap(depth, clip=2.0)

            out_frame = frame.copy()

            # Overlays
            if show_depth and last_depth_cm is not None:
                if overlay_mode == "blend":
                    out_frame = blend_depth(out_frame, last_depth_cm, alpha=args.alpha)
                elif overlay_mode == "side":
                    # side-by-side: original | depth
                    depth_vis = cv2.resize(last_depth_cm, (W, H), interpolation=cv2.INTER_LINEAR)
                    out_frame = np.hstack([out_frame, depth_vis])

            if show_boxes:
                draw_boxes(out_frame[:, :W], boxes, depth=last_depth, depth_near=args.depth_near, depth_far=args.depth_far)  # draw on left panel if side-by-side

            # HUD
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps

            if show_hud:
                put_text(out_frame, f"FPS {smoothed_fps:5.1f}", (10, 22))
                if yolo_times:
                    put_text(out_frame, f"YOLO {np.mean(yolo_times)*1000:5.1f} ms", (10, 42))
                if depth_times:
                    depth_freq = "live" if args.live_depth else f"every {depth_stride}"
                    put_text(out_frame, f"Depth {np.mean(depth_times)*1000:5.1f} ms ({depth_freq})", (10, 62))
                put_text(out_frame, f"Overlay: {overlay_mode}  Boxes: {'on' if show_boxes else 'off'}  Depth: {'on' if show_depth else 'off'}",
                         (10, 82))
                # Add frame counter and depth update indicator
                put_text(out_frame, f"Frame: {frame_idx}  Depth updated: {do_depth if 'do_depth' in locals() else 'N/A'}", (10, 102))

            # Write & show
            if writer is not None:
                writer.write(out_frame)
            cv2.imshow(window_name, out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_depth = not show_depth
            elif key == ord('m'):
                overlay_mode = "side" if overlay_mode == "blend" else "blend"
            elif key == ord('b'):
                show_boxes = not show_boxes
            elif key == ord('i'):
                show_hud = not show_hud
            elif key == ord('r'):
                if writer is None:
                    out_path = Path("capture.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (out_frame.shape[1], out_frame.shape[0]))
                    if writer.isOpened():
                        print(f"[Rec] Recording to {out_path.resolve()}")
                    else:
                        print("[Rec] Could not start recording.")
                        writer = None
                else:
                    print("[Rec] Stopped.")
                    writer.release()
                    writer = None

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
