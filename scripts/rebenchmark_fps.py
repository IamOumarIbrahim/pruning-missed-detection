"""Controlled FPS re-benchmarking script (zero contention, single-pass)."""

import json, time, glob, os, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import prune.pruner
import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def benchmark_pt(model_path, imgsz=640, device='cuda:0', warmup=20, iters=50):
    m = YOLO(model_path).model.to(device)
    m.eval()
    x = torch.zeros((1, 3, imgsz, imgsz), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = m(x)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / iters * 1000.0
        fps = 1000.0 / latency_ms
    del m
    torch.cuda.empty_cache()
    return round(fps, 1), round(latency_ms, 2)

def benchmark_onnx(model_path, imgsz=640, warmup=10, iters=30):
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), sess_opt, providers=['CPUExecutionProvider'])
    inp_obj = sess.get_inputs()[0]
    inp = inp_obj.name
    out = sess.get_outputs()[0].name
    dtype = np.float16 if 'float16' in inp_obj.type else np.float32
    x = np.zeros((1, 3, imgsz, imgsz), dtype=dtype)
    for _ in range(warmup):
        _ = sess.run([out], {inp: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run([out], {inp: x})
    latency_ms = (time.perf_counter() - t0) / iters * 1000.0
    fps = 1000.0 / latency_ms
    del sess
    return round(fps, 1), round(latency_ms, 2)

def main():
    files = sorted(glob.glob('results/**/metrics.json', recursive=True))
    total = len(files)
    print(f'Starting controlled FPS re-benchmarking on {total} configurations...')

    for idx, f in enumerate(files, 1):
        p = Path(f)
        with open(p) as fp:
            d = json.load(fp)

        # Determine model path
        model_path = d.get('engine_path') or d.get('weights_path')
        if not model_path or not Path(model_path).exists():
            # Check alternative extensions
            candidate_paths = [
                p.parent / 'weights' / 'best.pt',
                Path(str(model_path).replace('.pt', '.onnx')) if model_path else None,
                Path(str(model_path).replace('.engine', '.onnx')) if model_path else None,
            ]
            for cand in candidate_paths:
                if cand and cand.exists():
                    model_path = str(cand)
                    break

        if not model_path or not Path(model_path).exists():
            print(f'[{idx}/{total}] Skip: could not find model file for {f}')
            continue

        model_path = Path(model_path)
        is_onnx = model_path.suffix == '.onnx'

        try:
            if is_onnx:
                fps, latency = benchmark_onnx(model_path)
            else:
                fps, latency = benchmark_pt(model_path)

            d['fps'] = fps
            d['latency_ms'] = latency

            with open(p, 'w') as fp:
                json.dump(d, fp, indent=2)

            model_name = d.get('model', '')
            label = d.get('pruning_label', '')
            q = d.get('quantization', '')
            seed = d.get('seed', '')
            print(f'[{idx}/{total}] {model_name} {label} {q} seed {seed}: {latency} ms ({fps} FPS)')
        except Exception as e:
            print(f'[{idx}/{total}] ERROR {model_path}: {e}')

    print('Controlled FPS re-benchmarking complete.')

if __name__ == '__main__':
    main()
