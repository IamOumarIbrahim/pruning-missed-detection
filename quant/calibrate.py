"""Calibration set sampling for PTQ INT8.

Samples images from the training split (not val) as specified in the
README: 300 images, fixed seed.
"""

import random
from pathlib import Path
import yaml


def sample_calibration_set(data_yaml, num_samples=300, seed=42,
                           output_path=None):
    """Sample calibration images from the training split.

    Args:
        data_yaml: Path to Ultralytics dataset YAML.
        num_samples: Number of images to sample.
        seed: Random seed.
        output_path: Optional path to write the list.

    Returns:
        list of absolute image path strings.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(data_yaml).parent / cfg['path']
    train_file = data_root / cfg['train']

    with open(train_file) as f:
        all_images = [l.strip() for l in f if l.strip()]

    txt_dir = train_file.parent
    resolved = [(txt_dir / p).resolve() for p in all_images]

    rng = random.Random(seed)
    n = min(num_samples, len(resolved))
    sampled = rng.sample(resolved, n)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for p in sampled:
                f.write(str(p) + '\n')

    return [str(p) for p in sampled]
