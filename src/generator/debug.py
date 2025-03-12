import cv2
import os

from src.poisson_config import IF_DEBUG_FILES, DEBUG_FILES_PATH

def save_debug_img_pil(pil_image, debug_file_name, description=""):
    if not os.path.exists(DEBUG_FILES_PATH):
        os.makedirs(DEBUG_FILES_PATH)

    if IF_DEBUG_FILES:
        p = f"{DEBUG_FILES_PATH}{debug_file_name}{description}.png"
        pil_image.save(p)

def save_debug_img(pil_image_arr, debug_file_name, dirname, description=""):

    if IF_DEBUG_FILES:    
        if not os.path.exists(DEBUG_FILES_PATH):
            os.makedirs(DEBUG_FILES_PATH)

        numbered_path = os.path.join(DEBUG_FILES_PATH, dirname)
        if not os.path.exists(numbered_path):
            os.makedirs(numbered_path)

        p = os.path.join(numbered_path, f"{debug_file_name}{description}.png")
              
        if len(pil_image_arr.shape) == 2: 
            cv2.imwrite(p, pil_image_arr)
        else:
            if pil_image_arr.shape[2] == 3:
                cv2.imwrite(p, cv2.cvtColor(pil_image_arr, cv2.COLOR_RGB2BGR))
            elif pil_image_arr.shape[2] == 4:
                cv2.imwrite(p, cv2.cvtColor(pil_image_arr, cv2.COLOR_RGBA2BGRA))
            else:
                cv2.imwrite(p, pil_image_arr)