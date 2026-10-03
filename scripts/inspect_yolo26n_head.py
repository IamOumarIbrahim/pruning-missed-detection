from ultralytics import YOLO

print("="*70)
print("INSPECTING YOLO11n vs YOLO26n ARCHITECTURE & DETECT HEAD")
print("="*70)

for m in ['yolo11n', 'yolo26n']:
    model = YOLO(f"{m}.pt")
    head = model.model.model[-1]
    head_type = head.__class__.__name__
    print(f"\n--- {m} ---")
    print(f"Head Module Type: {head_type}")
    print(f"Has end2end attribute? {hasattr(head, 'end2end')}")
    if hasattr(head, 'end2end'):
        print(f"end2end value: {head.end2end}")
    print(f"Attributes: {dir(head)}")
    
    # Check validator / predictor NMS behavior
    val = model.val
    print(f"Default end2end in overrides: {model.overrides.get('end2end', 'N/A')}")
