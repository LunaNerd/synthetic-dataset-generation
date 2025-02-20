# Parameters for generator
NUMBER_OF_WORKERS = 16
BLENDING_LIST = [
    #"gaussian",
    "poisson",  # takes a lot of time and results are not that good
    # "poisson-fast",  # only with Docker GPU
    #"none",
    #"box",
    #"motion",     the motion blending method broke, is possible to copy old version from: 
    #              https://github.com/PD-Mera/pyblur/blob/master/pyblur/LinearMotionBlur.py and get it working
    #"mixed",
    #"illumination",
    #"gamma_correction",
]

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 4
MIN_NO_OF_DISTRACTOR_OBJECTS = 0
MAX_NO_OF_DISTRACTOR_OBJECTS = 0
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# OPTIONS:
#   GLOBAL = use MIN_SCALE and MAX_SCALE for all objects and distractors
#   PER_CLASS_STD = use std and mean per object class from inside complementary data, 
#                   
#   PER_CLASS_TRUNC_NORM = uses std and norm but limits to min and max value (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)
#
# if scale_augment = False inside generate_synthetic_data.py, no scaling will be applied.
AUGMENTATION_SIZE_OPTION = "PER_CLASS_TRUNC_NORM"
#AUGMENTATION_ROTATION_OPTION

# Parameters for objects in images
MIN_SCALE = 0.15  # min scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_SCALE = 0.4  # max scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_UPSCALING = 1.2  # increase size of foreground by max this
MAX_DEGREES = 180  # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = (
    0.25  # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
)
MAX_ALLOWED_IOU = 0.5  # IOU > MAX_ALLOWED_IOU is considered an occlusion, need dontocclude=True

# Parameters for image loading
MINFILTER_SIZE = 3

# Other

# Got replaced by OBJECT_CATEGORIES_PATH 
OBJECT_CATEGORIES = [
    { "id": 0, "name": "Ranunculus"},
    { "id": 1, "name": "Centaurea jacea"},
    { "id": 2, "name": "Senecio"},
    { "id": 3, "name": "Yellow composite"},
    { "id": 4, "name": "Hypericum perforatum"},
    { "id": 5, "name": "Rhinantus-like"},
    { "id": 6, "name": "Little/hop clover"},
    { "id": 7, "name": "Veronica"},
    { "id": 8, "name": "Trifolium pratense"},
    { "id": 9, "name": "Vicia"},
    { "id": 10, "name": "Eupatorium cannabinum"},
    { "id": 11, "name": "Daisy-like"},
    { "id": 12, "name": "Trifolium repens"},
    { "id": 13, "name": "Tanacetum vulgare"},
    { "id": 14, "name": "Geranium dissectum"},
    { "id": 15, "name": "Cirsium arvense"},
    { "id": 16, "name": "Umbellifer"},
    { "id": 17, "name": "Small white"},
    { "id": 18, "name": "Unknown"},
    { "id": 19, "name": "Symphytum officinale"},
    { "id": 20, "name": "Pulicaria dysenterica"},
    { "id": 21, "name": "Melilotus albus"},
    { "id": 22, "name": "Plantago lanceolata"}
]

DISTRACTOR_NAME = "distractor"
DISTRACTOR_ID = 2

IGNORE_LABELS = []  # list of category ID for which no annotations will be generated
INVERTED_MASK = False  # Set to true if white pixels represent background
SUPPORTED_IMG_FILE_TYPES = (".jpg", "jpeg", ".png", ".gif")

OBJECT_COMPLEMENTARY_DATA_PATH = "/project_ghent/luversmi/dataset/foreground/experiment_truncnorm/area_truncnorm_complementary_data.json"
OBJECT_CATEGORIES_PATH = "/project_ghent/luversmi/dataset/foreground/experiment_truncnorm/area_truncnorm_labels_1500_syn.json"

