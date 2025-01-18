import random

from PIL import Image

from src.config import MIN_SCALE, MAX_SCALE, MAX_UPSCALING


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
            break
    foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
    mask = mask.resize((o_w, o_h), Image.LANCZOS)
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

        print(scale)
        
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
