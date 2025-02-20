"""
pb: Poisson Image Blending implemented by Python
downloaded from [pb/pb.py at master · yskmt/pb](https://github.com/yskmt/pb/blob/master/pb.py)

MIT License: https://github.com/yskmt/pb/blob/master/LICENSE

"""

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import cv2
from random import randrange

from poisson_config import POISSON_MASK_STRATEGY, POISSON_BLEND_STRATEGY, IF_DEBUG_FILES, DEBUG_FILES_PATH

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

    mask = np.zeros((hm, wm))
    mask[img_mask > 0] = 1
    mask[img_mask == 0] = 0

    mask = mask[hd0:hd1, wd0:wd1]
    src = img_src[hd0:hd1, wd0:wd1]

    # fix offset
    offset_adj = (max(offset[0], 0), max(offset[1], 0))

    # remove edge from the mask so that we don't have to check the
    # edge condition
    mask[:, -1] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[0, :] = 0

    return mask, src, offset_adj


def get_gradient_sum(img, i, j, h, w):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    v_sum = (
        img[i, j] * 4 - img[i + 1, j] - img[i - 1, j] - img[i, j + 1] - img[i, j - 1]
    )

    return v_sum


def get_mixed_gradient_sum(img_src, img_target, i, j, h, w, ofs, c=1.0):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB

    c(>=0): larger, the more important the target image gradient is
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    nb = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for kk in range(4):
        fp = img_src[i, j] - img_src[i + nb[kk, 0], j + nb[kk, 1]]
        gp = (
            img_target[i + ofs[0], j + ofs[1]]
            - img_target[i + nb[kk, 0] + ofs[0], j + nb[kk, 1] + ofs[1]]
        )

        # if np.linalg.norm(fp) > np.linalg.norm(gp):
        #     v_sum += fp
        # else:
        #     v_sum += gp

        v_sum += np.array(
            [
                fp[0] if abs(fp[0] * c) > abs(gp[0]) else gp[0],
                fp[1] if abs(fp[1] * c) > abs(gp[1]) else gp[1],
                fp[2] if abs(fp[2] * c) > abs(gp[2]) else gp[2],
            ]
        )

    return v_sum

def save_debug_img(pil_image, debug_file_name, description=""):
    if IF_DEBUG_FILES:       
        p = f"{DEBUG_FILES_PATH}{debug_file_name}{description}.png"        
        if len(pil_image.shape) == 2: 
            cv2.imwrite(p, pil_image)
        else:
            if pil_image.shape[2] == 3:
                cv2.imwrite(p, cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR))
            elif pil_image.shape[2] == 4:
                cv2.imwrite(p, cv2.cvtColor(pil_image, cv2.COLOR_RGBA2BGRA))
            else:
                cv2.imwrite(p, pil_image)


#
# Failed idea (images are too compressed by the time they get here to create new usefull border imitations)
#
"""
def custom_copyMakeBorder(img, mask, pad_size):

    new_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    new_mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    height, width, _ = new_img.shape

    background_points = np.argwhere(new_mask==0)

    def find_nearest_point(points, given_point):
        #points = np.array(points)
        #given_point = np.array(given_point)
        # Calculate the Euclidean distance between the given point and all points in the list
        distances = np.linalg.norm(points - given_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = points[nearest_index]
        return nearest_point
    
    # Process the top and bottom edges
    for x in range(width):
        for y in range(pad_size):
            top_mask_pixel = new_mask[x, y]
            bottom_mask_pixel = new_mask[x, height - 1 - y]
            
            # Process top edge pixel (x, y)
            ### If pixel on border originates from inside the foreground mask, change it too...
            if top_mask_pixel == 255:
                p = find_nearest_point(background_points, [x, y])
                new_img[x, y] = new_img[p[0], p[1]]

            # Process bottom edge pixel (x, height - 1 - y)
            if bottom_mask_pixel == 255:
                p = find_nearest_point(background_points, [x, y])
                new_img[x, height - 1 - y] = new_img[p[0], p[1]]

    # Process the left and right edges
    for y in range(height):
        for x in range(pad_size):
            left_mask_pixel = new_mask[x, y]
            right_mask_pixel = new_mask[width - 1 - x, y]

            # Process left edge pixel (x, y)
            if left_mask_pixel == 255:
                p = find_nearest_point(background_points, [x, y])
                new_img[x, y] = new_img[p[0], p[1]]

            # Process right edge pixel (width - 1 - x, y)
            if right_mask_pixel == 255:
                p = find_nearest_point(background_points, [width - 1 - x, y])
                new_img[width - 1 - x, y] = new_img[p[0], p[1]]
    return new_img
"""
            
def poisson_blend(
    img_mask, img_src, img_target, method="mix", c=1.0, offset_adj=(0, 0), debug_file_name=""
):
    
    # ToDo: change img mask from np.float64 to 8 bit integers
    assert np.max(img_mask) < 1.00001 and np.min(img_mask) >= 0.0
    assert np.max(img_src) < 256 and np.max(img_src) >= 0

    new_img_src = np.uint8(img_src)
    new_img_mask = np.uint8(img_mask) * 255

    if debug_file_name:
        debug_file_name = debug_file_name.split(".")[0]
    else:
        debug_file_name = randrange(1000, 2000, 1)

    match POISSON_MASK_STRATEGY:
        case "RECTANGLE":
            new_img_mask = np.full_like(new_img_mask, 255, dtype=np.uint8)
            
        case "RECTANGLE+BORDER":
            pad_size = 5

            if new_img_mask[0][0] == 0:
                bg_px = new_img_src[0][0]
            elif new_img_mask[0][new_img_mask.shape[1]-1] == 0:
                bg_px = new_img_src[0][new_img_mask.shape[1]-1]
            elif new_img_mask[new_img_mask.shape[0]-1][new_img_mask.shape[1]-1] == 0:
                bg_px = new_img_src[new_img_mask.shape[0]-1][new_img_mask.shape[1]-1]
            elif new_img_mask[new_img_mask.shape[0]-1][0] == 0:
                bg_px = new_img_src[new_img_mask.shape[0]-1][0]
            else:
                print("Warning: none of the corner pixels where background, picking first background pixel in image")
                bg_px = new_img_src.where(new_img_mask == 0)[0]
                
            new_img_mask = np.full_like(new_img_mask, 255, dtype=np.uint8)
            new_img_mask = cv2.copyMakeBorder(new_img_mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=255)

            color = [int(bg_px[0]), int(bg_px[1]), int(bg_px[2])]

            new_img_src = cv2.copyMakeBorder(new_img_src, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, None, color)       
            
            save_debug_img(new_img_src, debug_file_name, description="_mean")

        case "DILATE":
            visualizing_results = False
            if visualizing_results:
                number = randrange(1000, 2000, 1)
            
            #if visualizing_results:
            #    cv2.imwrite(f"/project_ghent/luversmi/attempt2/test/{debug_file_name}_before.png", new_img_src)

            save_debug_img(new_img_src, debug_file_name, "_before")
            pad_size = 6
            
            # Unsuccessfull for now
            #padded_image = custom_copyMakeBorder(new_img_src, new_img_mask, pad_size)         
            
            #
            #
            masked_pixels = new_img_src[new_img_mask != 255]

            # Calculate the average color
            mean_color = masked_pixels.mean(axis=0)
            color = [mean_color[0], mean_color[1], mean_color[2]]
            #
            #

            padded_image = cv2.copyMakeBorder(new_img_src, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=color)
            padded_mask = cv2.copyMakeBorder(new_img_mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
            
            save_debug_img(padded_image, debug_file_name, "_border")
            save_debug_img(padded_mask, debug_file_name, "_mask_before")

            #cv2.imwrite(f"/project_ghent/luversmi/attempt2/test/{debug_file_name}_bordervb.png", padded_image)
                  
            kernel = np.ones((11,11),np.uint8)
            new_img_mask = cv2.dilate(padded_mask,kernel,iterations = 1)
            alpha_data = np.uint8(new_img_mask)

            save_debug_img(alpha_data, debug_file_name, "_mask_after")
            rgba = cv2.cvtColor(padded_image, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = alpha_data


            save_debug_img(rgba, debug_file_name, "_after")
            # if visualizing_results:
            #     cv2.imwrite(f"/project_ghent/luversmi/attempt2/test/{debug_file_name}_mask_after.png", rgba)

            new_img_src = padded_image

        case "ONLY_OBJECT":
            new_img_mask = np.uint8(img_mask * 255)
            
        case _:
            raise Exception("Invallid POISSON_MASK_STRATEGY")

    offset = [offset_adj[1] + int(round(img_src.shape[1] / 2)), offset_adj[0] + int(round(img_src.shape[0]) / 2)]
    
    shape = img_src.shape
    shape2 = img_mask.shape

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
        return output

    except Exception as e:
        #print(type(e))
        exception_border = np.max(new_img_mask.shape)

        save_debug_img(img_target, debug_file_name, "_exept_before")

        temp_img_target = cv2.copyMakeBorder(img_target, exception_border, exception_border, exception_border, exception_border, cv2.BORDER_REFLECT)
        
        output = cv2.seamlessClone(
            new_img_src, 
            temp_img_target, 
            new_img_mask, 
            [int(offset[0]+exception_border), int(offset[1]+exception_border)], 
            strategy
        )

        output = output[exception_border:output.shape[0]-exception_border, exception_border:output.shape[1]-exception_border, :]
        img_target = output.copy()
        save_debug_img(img_target, debug_file_name, "_exept_after")

        
        #print("Poisson-blending failed: \t reverting to 'None' method, this error is because flower overlaps with border background")
        return img_target

    # try:
    #     output = cv2.seamlessClone(
    #         new_img_src, 
    #         img_target, 
    #         new_img_mask, 
    #         [offset_adj[1] + int(round(img_src.shape[1] / 2)), offset_adj[0] + int(round(img_src.shape[0]) / 2)], 
    #         cv2.MIXED_CLONE_WIDE)
    #         #cv2.NORMAL_CLONE_WIDE)
    # except:
    #     print("...failed")
    #     return img_target
    
    return output
    # hm, wm = img_mask.shape
    # region_size = hm * wm

    # F = np.zeros((region_size, 3))
    # A = scipy.sparse.identity(region_size, format="lil")

    # get_k = lambda i, j: i + j * hm

    # # plane insertion
    # if method in ["target", "src"]:
    #     for i in range(hm):
    #         for j in range(wm):
    #             k = get_k(i, j)

    #             # ignore the edge case (# of neighboor is always 4)
    #             if img_mask[i, j] == 1:

    #                 if method == "target":
    #                     F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]
    #                 elif method == "src":
    #                     F[k] = img_src[i, j]
    #             else:
    #                 F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    # # poisson blending
    # else:
    #     if method == "mix":
    #         grad_func = lambda ii, jj: get_mixed_gradient_sum(
    #             img_src, img_target, ii, jj, hm, wm, offset_adj, c=c
    #         )
    #     else:
    #         grad_func = lambda ii, jj: get_gradient_sum(img_src, ii, jj, hm, wm)

    #     for i in range(hm):
    #         for j in range(wm):
    #             k = get_k(i, j)

    #             # ignore the edge case (# of neighboor is always 4)
    #             if img_mask[i, j] == 1:
    #                 f_star = np.array([0.0, 0.0, 0.0])

    #                 if img_mask[i - 1, j] == 1:
    #                     A[k, k - 1] = -1
    #                 else:
    #                     f_star += img_target[i - 1 + offset_adj[0], j + offset_adj[1]]

    #                 if img_mask[i + 1, j] == 1:
    #                     A[k, k + 1] = -1
    #                 else:
    #                     f_star += img_target[i + 1 + offset_adj[0], j + offset_adj[1]]

    #                 if img_mask[i, j - 1] == 1:
    #                     A[k, k - hm] = -1
    #                 else:
    #                     f_star += img_target[i + offset_adj[0], j - 1 + offset_adj[1]]

    #                 if img_mask[i, j + 1] == 1:
    #                     A[k, k + hm] = -1
    #                 else:
    #                     f_star += img_target[i + offset_adj[0], j + 1 + offset_adj[1]]

    #                 A[k, k] = 4
    #                 F[k] = grad_func(i, j) + f_star

    #             else:
    #                 F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    # A = A.tocsr()

    # img_pro = np.empty_like(img_target.astype(np.uint8))
    # img_pro[:] = img_target.astype(np.uint8)

    # for l in range(3):
    #     # x = pyamg.solve(A, F[:, l], verb=True, tol=1e-15, maxiter=100)
    #     x = spsolve(A, F[:, l])
    #     x[x > 255] = 255
    #     x[x < 0] = 0
    #     x = np.array(x, img_pro.dtype)

    #     img_pro[
    #         offset_adj[0] : offset_adj[0] + hm, offset_adj[1] : offset_adj[1] + wm, l
    #     ] = x.reshape(hm, wm, order="F")

    # return img_pro
