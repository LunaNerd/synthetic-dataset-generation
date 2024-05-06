# Parameters for generator
NUMBER_OF_WORKERS = 1
BLENDING_LIST = [
    "gaussian",
    #"poisson",  # takes a lot of time and results are not that good
    # "poisson-fast",  # only with Docker GPU
    "none",
    #"box",
    #"motion",
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

# Parameters for objects in images
#TARGET_SIZE_PER_CATEGORY = TRUE
# If DEFAULT_SCALING = False
MIN_SCALE = 0.05  # min scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_SCALE = 0.2  # max scale for scale augmentation (maximum extend in each direction, 1=same size as image)

MAX_UPSCALING = 0.7  # increase size of foreground by max this
MAX_DEGREES = 30  # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = (
    0.25  # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
)
MAX_ALLOWED_IOU = 0.5  # IOU > MAX_ALLOWED_IOU is considered an occlusion, need dontocclude=True

# Parameters for image loading
MINFILTER_SIZE = 3

# Other
OBJECT_CATEGORIES = [
	{'id': 17, 'name': '17'},
	{'id': 6, 'name': '6'},
	{'id': 1, 'name': '1'},
	{'id': 10, 'name': '10'},
	{'id': 22, 'name': '22'},
	{'id': 14, 'name': '14'},
	{'id': 5, 'name': '5'},
	{'id': 8, 'name': '8'},
	{'id': 2, 'name': '2'},
	{'id': 19, 'name': '19'},
	{'id': 13, 'name': '13'},
	{'id': 11, 'name': '11'},
	{'id': 0, 'name': '0'},
	{'id': 7, 'name': '7'},
	{'id': 16, 'name': '16'},
	{'id': 12, 'name': '12'},
	{'id': 20, 'name': '20'},
	{'id': 3, 'name': '3'},
	{'id': 9, 'name': '9'},
	{'id': 4, 'name': '4'},
	{'id': 21, 'name': '21'},
	{'id': 15, 'name': '15'}
]  # note: distractor needs to be second position
IGNORE_LABELS = []  # list of category ID for which no annotations will be generated

OBJECT_DOMAIN_KNOWLEDGE = {
12: {'min_rel_scale': 0.02222165436043843, 'max_rel_scale': 0.06282856812617668, 'max_obj_degrees': 180}, 
14: {'min_rel_scale': 0.010507888459438074, 'max_rel_scale': 0.020836361540561923, 'max_obj_degrees': 180}, 
0: {'min_rel_scale': 0.023636658366405276, 'max_rel_scale': 0.03963435583237769, 'max_obj_degrees': 180}, 
2: {'min_rel_scale': 0.0075, 'max_rel_scale': 0.08234238974394692, 'max_obj_degrees': 180}, 
18: {'min_rel_scale': 0.0075, 'max_rel_scale': 0.0848674754977985, 'max_obj_degrees': 180}, 
6: {'min_rel_scale': 0.0075, 'max_rel_scale': 0.021865145191864954, 'max_obj_degrees': 180}, 
22: {'min_rel_scale': 0.02239567793668057,'max_rel_scale': 0.04697829920617658, 'max_obj_degrees': 180},
3: {'min_rel_scale': 0.03424289123342825, 'max_rel_scale': 0.06460652267031505, 'max_obj_degrees': 180},
17: {'min_rel_scale': 0.010180288784003056, 'max_rel_scale': 0.01875775119649426, 'max_obj_degrees': 180},
15: {'min_rel_scale': 0.019062640041671252, 'max_rel_scale': 0.050331468066436846, 'max_obj_degrees': 180},
16: {'min_rel_scale': 0.039712987210649975, 'max_rel_scale': 0.11322953817006068, 'max_obj_degrees': 180},
5: {'min_rel_scale': 0.025584910234939997, 'max_rel_scale': 0.0646750704413885, 'max_obj_degrees': 180},
8: {'min_rel_scale': 0.016300917166475085, 'max_rel_scale': 0.025715520130236157, 'max_obj_degrees': 180},
9: {'min_rel_scale': 0.008422239479125345, 'max_rel_scale': 0.026460190272940772, 'max_obj_degrees': 180},
11: {'min_rel_scale': 0.035770194891925344, 'max_rel_scale': 0.07576791986217302, 'max_obj_degrees': 180},
7: {'min_rel_scale': 0.010024407767909813, 'max_rel_scale': 0.01606049734157924, 'max_obj_degrees': 180},
20: {'min_rel_scale': 0.027403580945944504, 'max_rel_scale': 0.04621993553757198, 'max_obj_degrees': 180},
10: {'min_rel_scale': 0.06863549815569717, 'max_rel_scale': 0.2238356685109695, 'max_obj_degrees': 180},
13: {'min_rel_scale': 0.03015771496887531, 'max_rel_scale': 0.09601011989351, 'max_obj_degrees': 180},
21: {'min_rel_scale': 0.03499521292390311, 'max_rel_scale': 0.11175185374276356, 'max_obj_degrees': 180},
19: {'min_rel_scale': 0.009328490097773284, 'max_rel_scale': 0.16834392312536722, 'max_obj_degrees': 180},
1: {'min_rel_scale': 0.017464678151861864, 'max_rel_scale': 0.03564153237445393, 'max_obj_degrees': 180},
4: {'min_rel_scale': 0.025381904328246167, 'max_rel_scale': 0.06497019090984907, 'max_obj_degrees': 180}}

INVERTED_MASK = False  # Set to true if white pixels represent background
SUPPORTED_IMG_FILE_TYPES = (".jpg", "jpeg", ".png", ".gif")
