import cv2
import os

from src.config import IF_DEBUG_FILES, DEBUG_FILES_PATH

class debug_config:
    def __init__(self, syn_img_nr="", object_file_name="", debug_img_dest_path=DEBUG_FILES_PATH):
        self.object_file_name = object_file_name
        self.syn_img_nr = syn_img_nr
        self.debug_img_dest_path = debug_img_dest_path

        if IF_DEBUG_FILES:
            temp_path = os.path.join(self.object_file_name, syn_img_nr)
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

    def __create_path__(self, description):
        return os.path.join(self.debug_img_dest_path, self.syn_img_nr, f"{self.object_file_name}{description}.png")

    def save_pil(pil_image, description=""):
        if IF_DEBUG_FILES:
            
            p = __create_path__(description)
            pil_image.save(p)

    def save(pil_image_arr, description=""):

        if IF_DEBUG_FILES:

            p = self.__create_path__(description)
                
            if len(pil_image_arr.shape) == 2: 
                cv2.imwrite(p, pil_image_arr)
            else:
                if pil_image_arr.shape[2] == 3:
                    cv2.imwrite(p, cv2.cvtColor(pil_image_arr, cv2.COLOR_RGB2BGR))
                elif pil_image_arr.shape[2] == 4:
                    cv2.imwrite(p, cv2.cvtColor(pil_image_arr, cv2.COLOR_RGBA2BGRA))
                else:
                    cv2.imwrite(p, pil_image_arr)