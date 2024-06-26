from pathlib import Path
import sys
import os

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())
import shutil
import random
import numpy as np
from src.generator.handler import generate_synthetic_dataset

seed = 42
random.seed(seed)
np.random.seed(seed)

DATA_DIR = Path("/project_ghent/luversmi/dataset/synthetic_data/experiment_flowering")


if __name__ == "__main__":
    dataset_name = "6250_only_syn"  # Give your dataset a name
    output_dir = (DATA_DIR / dataset_name).resolve()
    #output_dir = os.path.join(DATA_DIR, dataset_name)
    if output_dir.exists():
        shutil.rmtree(output_dir.as_posix())
    output_dir.mkdir()
    docker = False
    if not docker:
        # Adjust paths here if you are not using Docker
        
        distractor_json = "/project_ghent/luversmi/dataset/background/empty/empty_list.json"
        object_json = "/project_ghent/luversmi/dataset/foreground/experiment_flowering/foreground_1500_syn.json"
        background_json = "/project_ghent/luversmi/dataset/background/empty/with_rotation.json"
        
    else:
        distractor_json = "/data/distractors/splits.json"
        object_json = "/data/objects/splits.json"
        background_json = "/data/backgrounds/splits.json"
    generate_synthetic_dataset(
        output_dir=str(output_dir),
        object_json=str(object_json),
        distractor_json=str(distractor_json),
        background_json=str(background_json),
        number_of_images={
            "train": 6250,
            "validation": 0,
            "test": 0,
        },  # multiplied by blending methods,
        dontocclude=True,  # enable occlusion checking of objects
        rotation=True,  # enable random rotation of objects
        scale=True,  # enable random scaling of objects
        multithreading=True,  # enable multithreading for faster dataset generation
        label_json="/project_ghent/luversmi/dataset/foreground/experiment_flowering/labels_1500_syn.json"
    )
