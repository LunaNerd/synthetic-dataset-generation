from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())
import shutil
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

DATA_DIR = ROOT / "data"

config_path = ROOT / "src/config_demo_examples/config_truncnorm_poisson_005.py"
poisson_config_path = ROOT / "src/config_demo_examples/poisson_config_recommended.py"

print(poisson_config_path)

# https://galea.medium.com/symlink-use-cases-shortcuts-app-config-files-43b8ecf75a5
os.unlink(ROOT / "src/config.py")
os.symlink(config_path, ROOT / "src/config.py")

#os.unlink(ROOT / "src/poisson_config.py")
os.symlink(poisson_config_path, ROOT / "src/poisson_config.py")

from src.generator.handler import generate_synthetic_dataset

if __name__ == "__main__":
    
    dataset_name = "demo_truncnorm_poisson_005"  # Give your dataset a name
    output_dir = (DATA_DIR / dataset_name).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir.as_posix())
    output_dir.mkdir()

    distractor_json = ROOT / "data/distractors/splits.json"
    object_json = ROOT / "data/objects/splits.json"
    background_json = ROOT / "data/backgrounds/splits.json"

    generate_synthetic_dataset(
        output_dir=str(output_dir),
        object_json=str(object_json),
        distractor_json=str(distractor_json),
        background_json=str(background_json),
        number_of_images={
            "train": 5,
            "validation": 0,
            "test": 0,
        },  # multiplied by blending methods,
        dontocclude=True,  # enable occlusion checking of objects
        rotation=False,  # enable random rotation of objects
        scale=True,  # enable random scaling of objects
        multithreading=False,  # enable multithreading for faster dataset generation
    )
