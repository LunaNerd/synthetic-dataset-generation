"""
pb: Poisson Image Blending implemented by Python
downloaded from [pb/pb.py at master · yskmt/pb](https://github.com/yskmt/pb/blob/master/pb.py)
(only the create_mask function is used in this fork)

MIT License: https://github.com/yskmt/pb/blob/master/LICENSE

"""

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import cv2
from random import randrange

from src.poisson_config import POISSON_MASK_STRATEGY, POISSON_BLEND_STRATEGY, BORDER_SIZE_PX
from src.generator.debug import debug_config

def create_mask(img_mask, img_target, img_src, offset=(0, 0)):
    """
    Takes the np.array from the grayscale image
    """

    # crop img_mask and img_src to fit to the img_target
    hm, wm = img_mask.shape
    ht, wt, nl = img_target.shape

    hd0 = max(0, -offset[0])
    wd0 = max(0, -offset[1])

    hd1 = hm - max(hm + offset[0] - ht, 0)
    wd1 = wm - max(wm + offset[1] - wt, 0)

    # convert mask from [0, 255] to [0, 1] values
    # Not needed for this usecase
    """
    mask = np.zeros((hm, wm))
    mask[img_mask > 0] = 1
    mask[img_mask == 0] = 0
    """
    mask = np.copy(img_mask)

    mask = mask[hd0:hd1, wd0:wd1]
    src = img_src[hd0:hd1, wd0:wd1]

    # fix offset
    offset_adj = (max(offset[0], 0), max(offset[1], 0))

    # remove edge from the mask so that we don't have to check the
    # edge condition
    # Not needed for this implementation
    """
    mask[:, -1] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[0, :] = 0
    """
    return mask, src, offset_adj

def _treshold_mask(mask, treshold=0):
    cp_mask = mask.copy()
    mask[cp_mask > treshold] = 255
    mask[cp_mask <= treshold] = 0
    return cp_mask

def _calc_avg_color(img_target_patch):
    avg_color_patch = avg_color_patch.reshape(-1, 3)
    mean_target_color = avg_color_patch.mean(axis=0)
    mean_target_color = mean_target_color.astype(np.uint8)
    return mean_target_color


def _color_visualization(target_color, src_bg_color, debug_obj):
    image = np.zeros((300, 300, 3), np.uint8)

    # Fill image with red color(set each pixel to red)
    image[0:150, :, :] = (src_bg_color[0], src_bg_color1[0], src_bg_color[2])
    image[150:, :, :] = (target_color[0], target_color[1], target_color[2])

    debug_obj.save(image, "_target_color")

def _treshold_visualization(img_src, img_mask, debug_obj, treshold=0):
    debug_obj.save(img_mask, f"_mask_tr_{MASK_TRESHOLD}")
        
    test_img = img_src.copy()
    test_img[img_mask == 255] = 255
    debug_obj.save(test_img, f"_mask_visual_tr_{MASK_TRESHOLD}_1")

    test_img = img_src.copy()
    test_img[img_mask == 0] = 255
    debug_obj.save(test_img, f"_mask_visual_tr_{MASK_TRESHOLD}_2")

def poisson_blend(img_mask, img_src, img_target, offset_adj, bg_color, debug_obj):            
# def poisson_blend(
#     img_mask, img_src, img_target, method="mix", c=1.0, offset_adj=(0, 0), background_color=None, dirname="debug", debug_file_name=""
# ):
    
    # DONE: more strict asserts here
    # DONE: move color conversion here
    # DONE: create debug object
    # NOT NEEDED: use supplementary information more
    # TODO: TEST

    #assert np.max(img_mask) < 1.00001 and np.min(img_mask) >= 0.0
    assert img_src.dtype == np.uint8
    assert img_mask.dtype == np.uint8

    #new_img_src = np.uint8(img_src)
    #new_img_mask = np.uint8(img_mask)

    debug_obj.save(new_img_mask, "_mask_after_resize")
  
    MASK_TRESHOLD = 16
    new_img_mask = _treshold_mask(new_img_mask, MASK_TRESHOLD)
    _treshold_visualization(new_img_src, new_img_mask, debug_obj, treshold= MASK_TRESHOLD)

    bg_color = [int(bg_color[0]), int(bg_color[1]), int(bg_color[2])]

    #
    # If one wants to compare the impact of different treshold values: 
    #   un-comment this code and look at the debug images
    #
    # test_treshold = 0
    # test_img_mask = _treshold_mask(new_img_mask, test_treshold)
    # _treshold_visualization(new_img_src, test_img_mask, debug_obj, treshold= test_treshold)

    match POISSON_MASK_STRATEGY:
        case "RECTANGLE":
            new_img_mask = np.full_like(new_img_mask, 255, dtype=np.uint8)
            
        case "RECTANGLE+BORDER":
            # The true border size is still BORDER_SIZE_PX, 
            #       The +1 is to counteract mask trimming done by the OpenCV implementation of SeamlessClone
            pad_size = BORDER_SIZE_PX + 1 

            #bg_px = _find_background_pixel(new_img_mask, new_img_src)
            #bg_px = background_color
                
            new_img_mask = np.full_like(new_img_mask, 255, dtype=np.uint8)
            new_img_mask = cv2.copyMakeBorder(new_img_mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)

            #color = [int(bg_px[0]), int(bg_px[1]), int(bg_px[2])]

            new_img_src = cv2.copyMakeBorder(new_img_src, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, None, bg_color)       
            
            debug_obj.save(new_img_src, description="_mean")

        case "DILATE":
           
            #if visualizing_results:
            #    cv2.imwrite(f"/project_ghent/luversmi/attempt2/test/{debug_file_name}_before.png", new_img_src)

            debug_obj.save(new_img_src, description="_before")
            pad_size = BORDER_SIZE_PX        

            # Calculate the average color
            # mean_color = background_color
            # color = [int(mean_color[0]), int(mean_color[1]), int(mean_color[2])]

            padded_image = cv2.copyMakeBorder(new_img_src, pad_size+1, pad_size+1, pad_size+1, pad_size+1, cv2.BORDER_CONSTANT, None, value=bg_color)
            padded_mask = cv2.copyMakeBorder(new_img_mask, pad_size+1, pad_size+1, pad_size+1, pad_size+1, cv2.BORDER_CONSTANT, value=0)
            
            debug_obj.save(padded_image, "_border")
            debug_obj.save(padded_mask, "_mask_before")

            kernel_size = BORDER_SIZE_PX * 2 + 1    
            kernel = np.ones((kernel_size,kernel_size),np.uint8)
            new_img_mask = cv2.dilate(padded_mask,kernel,iterations = 1)
            alpha_data = np.uint8(new_img_mask)

            debug_obj.save(alpha_data, "_mask_after")
            rgba = cv2.cvtColor(padded_image, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = alpha_data

            debug_obj.save(rgba,"_after")

            new_img_src = padded_image

        case "ONLY_OBJECT":
            new_img_mask = np.uint8(img_mask * 255)
            
        case _:
            raise Exception("Invallid POISSON_MASK_STRATEGY")

    # offset needs to be adjusted so it is the center of the src image instead of the left top corner
    offset = [offset_adj[1] + int(round(img_src.shape[1] / 2)), offset_adj[0] + int(round(img_src.shape[0]) / 2)]

    slice_1 = slice(offset_adj[1], offset_adj[1] + img_src.shape[1], 1)
    slice_0 = slice(offset_adj[0], offset_adj[0] + img_src.shape[0], 1)

    #avg_color_patch = img_target[slice_1, slice_0, :]
    avg_color_patch = img_target[slice_0, slice_1, :]
    debug_obj.save(avg_color_patch, "_target_patch")

    avg_target_color = _calc_avg_color(avg_color_patch)
    _color_visualization(avg_target_color, bg_color, debug_obj)

    match POISSON_BLEND_STRATEGY:
        case "NORMAL_WIDE":
            strategy = cv2.NORMAL_CLONE_WIDE
        case "MIXED_WIDE":
            strategy = cv2.MIXED_CLONE_WIDE
        case _:
            raise Exception("invalid POISSON_BLEND_STRATEGY")
    
    try:
        output = cv2.seamlessClone(
            new_img_src, 
            img_target, 
            new_img_mask, 
            offset, 
            strategy
        )

        debug_obj.save(output[slice_0, slice_1, :], "_target_patch_after")
        return output

    except Exception as e:
        
        print(type(e))
        exception_border = np.max(new_img_mask.shape)

        debug_obj.save(img_target, "_exept_before")

        temp_img_target = cv2.copyMakeBorder(img_target, exception_border, exception_border, exception_border, exception_border, cv2.BORDER_REFLECT)
        
        output = cv2.seamlessClone(
            new_img_src, 
            temp_img_target, 
            new_img_mask, 
            [int(offset[0]+exception_border), int(offset[1]+exception_border)], 
            strategy
        )

        output = output[exception_border:output.shape[0]-exception_border, exception_border:output.shape[1]-exception_border, :]

        debug_obj.save(output[slice_0, slice_1, :], "_target_patch_after")
        #img_target = output.copy()
        debug_obj.save(output, debug_file_name, dirname, "_exept_after")

        return output