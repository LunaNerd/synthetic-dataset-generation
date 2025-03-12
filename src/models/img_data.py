from pathlib import Path

import json
import cv2
import numpy as np
from PIL import Image, ImageFilter

import os

from src.config import INVERTED_MASK, MINFILTER_SIZE
from src.config import OBJECT_CATEGORIES, OBJECT_COMPLEMENTARY_DATA_PATH

from src.generator.debug import save_debug_img, save_debug_img_pil

from src.poisson_config import POISSON_BACKGROUND_STRATEGY

class BaseImgData:
    # (Class variable, behaves different from object variable, is shared acros all objects
    complementary_data = None
    
    def __init__(self, img_path: Path, label_id: str, label = None):
        self.img_path = img_path
        self.label_id = label_id
        if not label:
            self.label = [f["name"] for f in OBJECT_CATEGORIES if label_id == f["id"]][0]
        else:
            self.label = label
        #self.load_complementary_data()

    def __str__(self):
        return f"Label {self.label} from {self.img_path}"

    def load_complementary_data(self):
        raise NotImplementedError("Should be implemented by subclass")

    def get_mask(self, opencv=True):
        raise NotImplementedError("Should be implemented by subclass")

    def get_image(self, opencv=True):
        if opencv:
            img = cv2.imread(self.img_path.as_posix())
        else:
            img = Image.open(self.img_path.as_posix())

        # check_tensor(to_numpy_image(img), 'h w 3')
        return img

    def get_annotation_from_mask(self, scale=1.0):
        """Given a mask file and scale, return the bounding box annotations

        Args:
        Returns:
            tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
        """
        mask = self.get_mask(opencv=True)
        if mask is not None:
            if INVERTED_MASK:
                mask = 255 - mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if len(np.where(rows)[0]) > 0:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                return (
                    int(scale * xmin),
                    int(scale * xmax),
                    int(scale * ymin),
                    int(scale * ymax),
                )
            else:
                return -1, -1, -1, -1
        else:
            print("Mask not found. Using empty mask instead.")
            return -1, -1, -1, -1

    def load_object_data(self):
        foreground = self.get_image(opencv=False)
        if foreground is None:
            return None
        xmin, xmax, ymin, ymax = self.get_annotation_from_mask()
        foreground = foreground.crop((xmin, ymin, xmax, ymax))
        orig_w, orig_h = foreground.size
        mask = self.get_mask(opencv=False)
        if mask is None:
            return None
        mask = mask.crop((xmin, ymin, xmax, ymax))
        return foreground, mask, orig_h, orig_w


class ImgDataRGBA(BaseImgData):
    def __init__(self, img_path: Path, label_id, label=None):
        super().__init__(img_path, label_id, label=label)
        self.bg_color = None

    def load_complementary_data(self):
        if not ImgDataRGBA.complementary_data:
            if isinstance(OBJECT_COMPLEMENTARY_DATA_PATH, str):
                json_file = Path(OBJECT_COMPLEMENTARY_DATA_PATH)
            else:
                json_file = OBJECT_COMPLEMENTARY_DATA_PATH
            assert json_file.exists(), f"File {json_file.resolve()} does not exist!"
            with open(json_file) as json_f:
                ImgDataRGBA.complementary_data = json.load(json_f)
        return ImgDataRGBA.complementary_data.get(self.label, None)


    def get_mask(self, opencv=False):
        with open(self.img_path.as_posix(), "rb") as f:
            image = Image.open(f)

            img_name = os.path.basename(self.img_path).split(".png")[0]
            
            #save_debug_img_pil(image.split()[3], img_name, "_get_mask_before" )
            if image.mode == "RGBA":
                mask = image.split()[3].filter(
                    ImageFilter.MinFilter(MINFILTER_SIZE)
                )  # MinFilter better than threshold
                #save_debug_img_pil(mask, img_name, "_get_mask_after")
                if opencv:
                    mask = np.asarray(mask).astype(np.uint8)
            else:
                mask = None
        return mask

    def get_image(self, opencv=False):
        with open(self.img_path.as_posix(), "rb") as f:
            img = Image.open(f)
            if opencv:
                raise Exception("Not Implemented in this fork")
                new_img = np.asarray(img).astype(np.uint8)[:, :, :3]
                # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            else:
                if img.mode == "RGBA": 

                    img_3 = img.convert('RGB')

                    image_np = np.array(img_3)
                    mask_np = np.array(img.split()[3])

                    #print(image_np.shape)

                    # Get the pixels outside the mask
                    masked_pixels = image_np[mask_np != 255]
                    # print("masked: ")
                    # print(masked_pixels.shape)
                    # print()

                    # Calculate the average color
                    mean_color = masked_pixels.mean(axis=0)

                    self.bg_color = tuple(mean_color.astype(np.uint8))

                    if POISSON_BACKGROUND_STRATEGY == 'ORIGINAL':
                        # This ignores the A channel, resulting in the original background appearing again
                        return img_3
                        
                    elif POISSON_BACKGROUND_STRATEGY == 'MEAN':
                        new_img = Image.new(
                            "RGB", img.size, self.bg_color
                        )  # even background

                        new_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                        return new_img

                    #elif POISSON_BACKGROUND_STRATEGY == 'WHITE':
                    #    self.bg_color = (255, 255, 255)
                    
                    else:
                        raise Exception("invalid POISSON_BACKGROUND_STATEGY")

                else:
                    print(f"No RGBA channel found for {self.img_path}")
                    return None
        return new_img

    def get_bg_color(self):

        return self.bg_color