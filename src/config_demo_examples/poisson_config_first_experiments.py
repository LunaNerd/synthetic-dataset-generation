#
# With this configuration you'll get only the flowers pasted onto the background. 
# The background of the flowers gets replaced with an average of all the colors of this background.
# Because mixed poisson blending compares the pattern of both foreground and background, and the background of the foreground is even, 
# always the background is chosen making the border dissapear.
#
# This method works well on small flowers with busy patterns, but on big flowers with big surfices without a pattern the background 
# peeks through too much
#


# options: "DILATE", "RECTANGLE", "ONLY_OBJECT", "RECTANGLE+BORDER"
POISSON_MASK_STRATEGY = "RECTANGLE+BORDER"

# options: "MIXED_WIDE", "NORMAL_WIDE"
POISSON_BLEND_STRATEGY = "MIXED_WIDE"

# size of of border added around object in pixels
BORDER_SIZE_PX = 5

# Whether to create debug files that show the different steps of the poisson blending:
IF_DEBUG_FILES = True
DEBUG_FILES_PATH = "/project_ghent/luversmi/attempt2/test_exept_fix/"

# options: "ORIGINAL", "MEAN"
POISSON_BACKGROUND_STRATEGY = "MEAN"