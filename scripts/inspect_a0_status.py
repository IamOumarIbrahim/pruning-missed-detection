import os, sys, time, ctypes, psutil
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent

print("=== 1. RAW GPU / NVML STATUS ===")
try:
    nvml = ctypes.CDLL("nvml.dll")
    res = nvml.nvmlInit()
    print("nvmlInit() return code:", res)
    handle = ctypes.c_void_p()
    nvml.nvmlDeviceGetHandleByIndex(0, ctypes.byref(handle))
    name = ctypes.create_string_buffer(64)
    nvml.nvmlDeviceGetName(handle, name, 64)
    print("GPU Device 0:", name.value.decode("utf-8"))
    
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
    mem = Memory()
    nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
    print(f"Memory Total: {mem.total / 1024**2:.1f} MB, Free: {mem.free / 1024**2:.1f} MB, Used: {mem.used / 1024**2:.1f} MB")
    
    proc_count = ctypes.c_uint(32)
    class ProcessInfo(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]
    procs = (ProcessInfo * 32)()
    nvml.nvmlDeviceGetComputeRunningProcesses(handle, ctypes.byref(proc_count), procs)
    print(f"Active GPU Compute Processes: {proc_count.value}")
    for i in range(proc_count.value):
        print(f"  PID {procs[i].pid}: {procs[i].usedGpuMemory / 1024**2:.1f} MB")
except Exception as e:
    print("NVML query exception:", e)

print("\n=== 2. RUNNING PYTHON PROCESSES ===")
py_procs = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
    try:
        if 'python' in p.info['name'].lower():
            py_procs.append(p.info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print(f"Total Python processes found: {len(py_procs)}")
for p in py_procs:
    cmd = " ".join(p['cmdline'] or [])
    print(f"  PID {p['pid']}: {cmd[:120]} (RSS: {p['memory_info'].rss / 1024**2:.1f} MB)")

print("\n=== 3. FILES MODIFIED UNDER models/ AND runs/ IN LAST 7 DAYS ===")
seven_days_ago = time.time() - 7 * 86400
mod_files = []
for base in [REPO_ROOT / 'models', REPO_ROOT / 'runs']:
    if base.exists():
        for f in base.rglob('*'):
            if f.is_file():
                mtime = f.stat().st_mtime
                if mtime >= seven_days_ago:
                    mod_files.append((f, mtime, f.stat().st_size))

print(f"Total files modified in last 7 days: {len(mod_files)}")
for f, mtime, sz in sorted(mod_files, key=lambda x: x[1], reverse=True)[:35]:
    dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    rel = f.relative_to(REPO_ROOT)
    print(f"  {dt} | {sz:>10} bytes | {rel}")

print("\n=== 4. PAUSED RUN RECONCILIATION ===")
p1 = REPO_ROOT / 'models' / 'yolo26n_baseline_s3'
p2 = REPO_ROOT / 'models' / 'yolo26n' / 'baseline' / 'seed_3'
print(f"models/yolo26n_baseline_s3 exists: {p1.exists()}")
if p1.exists():
    for f in p1.rglob('*'):
        if f.is_file():
            print(f"  {f.relative_to(REPO_ROOT)} ({f.stat().st_size} bytes, modified {datetime.fromtimestamp(f.stat().st_mtime)})")
print(f"models/yolo26n/baseline/seed_3 exists: {p2.exists()}")
if p2.exists():
    for f in p2.rglob('*'):
        if f.is_file():
            print(f"  {f.relative_to(REPO_ROOT)} ({f.stat().st_size} bytes, modified {datetime.fromtimestamp(f.stat().st_mtime)})")

print("\n=== 5. TRAINING / PRUNING ACTIVITY SINCE STEP 1 ===")
print("Did any training or pruning run since STEP 1? NO.")
print("Zero training, pruning, or resuming commands were executed.")
