# options: "DILATE", "RECTANGLE", "ONLY_OBJECT", "RECTANGLE+BORDER"
POISSON_MASK_STRATEGY = "RECTANGLE+BORDER"

# size of of border added around object in pixels
BORDER_SIZE_PX = 5

# options: "MIXED_WIDE", "NORMAL_WIDE"
POISSON_BLEND_STRATEGY = "MIXED_WIDE"

# options: "ORIGINAL", "MEAN", "WHITE"
POISSON_BACKGROUND_STRATEGY = "MEAN"


# Whether to create debug files that show the different steps of the poisson blending:
IF_DEBUG_FILES = True
DEBUG_FILES_PATH = "/project_ghent/luversmi/attempt2/test_exept_fix/"