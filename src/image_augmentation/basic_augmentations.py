import random

from PIL import Image

from src.config import MIN_SCALE, MAX_SCALE, MAX_UPSCALING
from scipy.stats import truncnorm

import math


def augment_rotation(foreground, h, mask, max_degrees, w):
    while True:
        rot_degrees = random.randint(-max_degrees, max_degrees)
        foreground_tmp = foreground.rotate(rot_degrees, expand=True)
        mask_tmp = mask.rotate(rot_degrees, expand=True)
        o_w, o_h = foreground_tmp.size
        if w - o_w > 0 and h - o_h > 0:
            break
    mask = mask_tmp
    foreground = foreground_tmp
    return foreground, mask, o_h, o_w


def augment_scale(foreground, bg_h, mask, fg_h, fg_w, bg_w):
    width_scale = fg_w / bg_w
    height_scale = fg_h / bg_h
    choosen_scale = max(
        width_scale, height_scale
    )  # scale between foreground and background
    while True:
        scale = random.uniform(MIN_SCALE, MAX_SCALE) * (1 / choosen_scale)
        scale = min(
            scale, MAX_UPSCALING
        )  # allow only certain upscaling to prevent blurry foregrounds
        o_w, o_h = int(scale * fg_w), int(scale * fg_h)
        if bg_w - o_w > 0 and bg_h - o_h > 0 and o_w > 0 and o_h > 0:
            foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
            mask = mask.resize((o_w, o_h), Image.LANCZOS) 
            break
        else:
            print("\tWarning: image scaling had to be redone because foreground was bigger then background, if you see this warning often, concider reducing the max scale of foreground objects")
        

    return foreground, mask, o_h, o_w

def augment_scale_std(foreground, bg_h, mask, fg_h, fg_w, bg_w, complementary_data):
    mean_rel_size = complementary_data["mean_rel_size"]
    std_rel_size = complementary_data["std_rel_size"]
    min_rel_size = complementary_data["min_rel_size"]
    max_rel_size = complementary_data["max_rel_size"]
    
    while True:
        scale = random.gauss(mean_rel_size, std_rel_size)
        scale = max(scale, min_rel_size)
        scale = min(scale, max_rel_size)

        wh_ratio = fg_w / fg_h

        # wanted_w_pixels = max(bg_w * scale, min_rel_size)
        # wanted_w_pixels = min(bg_w * scale, max_rel_size)

        # wanted_h_pixels = max(bg_h * scale, min_pixels)
        # wanted_h_pixels = min(bg_h * scale, min_pixels)

        wanted_h_pixels = bg_h * scale
        wanted_w_pixels = bg_w * scale

        #print(scale)
        
        o_w = None
        o_h = None

        if fg_w > fg_h:
            o_h = wanted_h_pixels
            o_w = int(o_h * wh_ratio)
            o_h = int(o_h)

        else:
            #fg_w <= fg_h
            o_w = wanted_w_pixels
            o_h = int(o_w / wh_ratio)
            o_w = int(o_w)

        if bg_w - o_w > 0 and bg_h - o_h > 0 and o_w > 0 and o_h > 0:
            break

    foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
    mask = mask.resize((o_w, o_h), Image.LANCZOS)

    return foreground, mask, o_h, o_w

def augment_scale_area_truncnorm(foreground, bg_h, mask, fg_h, fg_w, bg_w, complementary_data):
    mean_area = complementary_data["mean_area"]
    std_area = complementary_data["std_area"]
    min_area = complementary_data["min_area"]
    max_area = complementary_data["max_area"]

    while True:
        orig_area = fg_h * fg_w
        a, b = (min_area - mean_area) / std_area, (max_area - mean_area) / std_area
    
        new_area = truncnorm.rvs(a, b, loc=mean_area, scale=std_area, size=1)[0]#, random_state=None)
    
        #new_h = round(  fg_h * ( new_area / orig_area )  )
        #new_w = round(  fg_w * ( new_area / orig_area )  )

        # https://math.stackexchange.com/questions/3983228/how-to-resize-a-rectangle-to-a-specific-area-while-maintaining-the-aspect-ratio

        new_h = round(  math.sqrt ( new_area * fg_h / fg_w))
        new_w = round(  math.sqrt ( new_area * fg_w / fg_h))

        if bg_w - new_w > 0 and bg_h - new_h > 0 and new_w > 0 and new_h > 0:
            break
        
    foreground = foreground.resize((new_w, new_h), Image.LANCZOS)
    mask = mask.resize((new_w, new_h), Image.LANCZOS)
    
    return foreground, mask, new_h, new_w