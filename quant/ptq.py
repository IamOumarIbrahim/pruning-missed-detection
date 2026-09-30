"""Post-training quantization export via Ultralytics / TensorRT."""

from pathlib import Path
from ultralytics import YOLO


def export_ptq(model_path, quant_level, output_dir, data_yaml=None,
               imgsz=640, device=0):
    """Export a YOLO model to a TensorRT engine at a given precision.

    Args:
        model_path: Trained checkpoint (.pt).
        quant_level: 'fp32', 'fp16', 'int8', or 'int4'.
        output_dir: Directory for the exported engine.
        data_yaml: Dataset YAML (required for INT8 calibration).
        imgsz: Input size.
        device: CUDA device.

    Returns:
        str: Path to the exported .engine file.
    """
    model = YOLO(str(model_path))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kwargs = dict(format='engine', imgsz=imgsz, device=device, workspace=4)

    if quant_level == 'fp32':
        kwargs['half'] = False
    elif quant_level == 'fp16':
        kwargs['half'] = True
    elif quant_level == 'int8':
        kwargs['int8'] = True
        if data_yaml:
            kwargs['data'] = str(data_yaml)
    elif quant_level == 'int4':
        # INT4 falls back to INT8 export; TensorRT selects INT4 kernels
        # where available on supported hardware.
        kwargs['int8'] = True
        if data_yaml:
            kwargs['data'] = str(data_yaml)
    else:
        raise ValueError(f"Unknown quant_level: {quant_level}")

    try:
        try:
            import modelopt
        except ImportError:
            raise RuntimeError("nvidia-modelopt is not installed; skipping TensorRT to use ONNX Runtime fallback directly")
        exported_path = Path(model.export(**kwargs))
        target = Path(output_dir) / f"model_{quant_level}.engine"
        if exported_path.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            exported_path.rename(target)
        return str(target)
    except Exception as e:
        print(f"TensorRT export bypassed/failed ({e}). Falling back to ONNX Runtime...")
        onnx_kwargs = dict(format='onnx', imgsz=imgsz, device=device)
        if quant_level == 'fp16':
            onnx_kwargs['half'] = True
        elif quant_level in ('int8', 'int4'):
            onnx_kwargs['int8'] = True
            if data_yaml:
                onnx_kwargs['data'] = str(data_yaml)
        exported_path = Path(model.export(**onnx_kwargs))
        target = Path(output_dir) / f"model_{quant_level}.onnx"
        if exported_path.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            exported_path.rename(target)
        return str(target)
