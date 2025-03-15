from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())
import shutil
import random
import numpy as np
from src.generator.handler import generate_synthetic_dataset

seed = 42
random.seed(seed)
np.random.seed(seed)

DATA_DIR = Path("/project_ghent/luversmi/dataset/synthetic_data/experiment_truncnorm_poisson_det")

if __name__ == "__main__":
    dataset_name =  "coco_truncnorm_200_normal_wide_5px"  # Give your dataset a name
    output_dir = (DATA_DIR / dataset_name).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir.as_posix())
    output_dir.mkdir()

    distractor_json = "/project_ghent/luversmi/dataset/background/empty/empty_list.json"
    object_json = "/project_ghent/luversmi/dataset/foreground/experiment_flowering/foreground_1500_syn.json"
    background_json = "/project_ghent/luversmi/dataset/background/empty/with_rotation.json"

    generate_synthetic_dataset(
        output_dir=str(output_dir),
        object_json=str(object_json),
        distractor_json=str(distractor_json),
        background_json=str(background_json),
        number_of_images={
            "train": 200,
            "validation": 0,
            "test": 0,
        },  # multiplied by blending methods,
        dontocclude=True,  # enable occlusion checking of objects
        rotation=False,  # enable random rotation of objects
        # Rotation currently not tested and thus expect unexpected results
        scale=True,  # enable random scaling of objects
        multithreading=False,  # enable multithreading for faster dataset generation
    )
