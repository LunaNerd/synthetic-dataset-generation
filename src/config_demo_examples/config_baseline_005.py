# Parameters for generator
NUMBER_OF_WORKERS = 1
BLENDING_LIST = [
    "gaussian",
    #"poisson",
    "none",
    "box",
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
#   PER_CLASS_RAND_RANGE = use a seperate MIN_SCALE and MAX_SCALE for each object class
#   PER_CLASS_STD = use std and mean per object class from inside complementary data, 
#                   
#   PER_CLASS_TRUNC_NORM = uses std and norm but limits to min and max value (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)
#
# if scale_augment = False inside generate_synthetic_data.py, no scaling will be applied.
AUGMENTATION_SIZE_OPTION = "PER_CLASS_RAND_RANGE"

# Parameters for objects in images
MIN_SCALE = 0.15  # min scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_SCALE = 0.4  # max scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_UPSCALING = 1.2  # increase size of foregrtound by max this
MAX_DEGREES = 180  # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = (
    0.25  # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
)
MAX_ALLOWED_IOU = 0.5  # IOU > MAX_ALLOWED_IOU is considered an occlusion, need dontocclude=True

# Parameters for image loading
MINFILTER_SIZE = 3

# Other

OBJECT_CATEGORIES = [
    {"id": 0, "name": "box_type_0"},
    {"id": 1, "name": "box_type_1"},
]

DISTRACTOR_NAME = "distractor"
DISTRACTOR_ID = 2

IGNORE_LABELS = []  # list of category ID for which no annotations will be generated
INVERTED_MASK = False  # Set to true if white pixels represent background
SUPPORTED_IMG_FILE_TYPES = (".jpg", "jpeg", ".png", ".gif")

OBJECT_COMPLEMENTARY_DATA_PATH = "data/complementary_data_demo_examples/demo_complementary_data_min_max.json"
OBJECT_CATEGORIES_PATH = "data/objects/splits_labels.json"

IF_DEBUG_FILES = False
DEBUG_FILES_PATH = ""