import cv2
import numpy as np
import os
import multiprocessing
from decorators import function_timer
import shutil
import os

import NN_Process

#note! Images should be even# length & width for math to work.
#for textures, you need powers of 2. that means all resolution increases must be 2x,4x,8x,16x,32x, etc
r"""
Folders to Avoid
    water
    lod
    ps3, ps3o, xbox, french, german, italian, spanish
    nv_playingcards
Files to avoid:
    theres other effects, will need to reconcile manually.
    nvl38reflect_e.dds - X:\GOG\Fallout New Vegas\Data\textures\effects\nv
    terrainnoise
problem_images:
    Mentats
    kings outfit
    bossscribe outfit
"""

class NN_Resize():
    def __init__(self):
        self.transparency_mask = []
        self.parent_source_dir = ""
        self.out_dir = ""
        self.avoid_list = []
        self.i = 0
        self.model = ""
        self.forbidden_folders = ["water", "normals", "ps3", "ps3o", "xbox", "french", "german", "italian", "spanish", "nv_playingcards"]
        #self.forbidden_folders = []
        self.forbidden_regex = "_n.png" #let's not process negatives.
        
    #@function_timer
    def AI_resize_1(self, input_img, model="", mask_bool=True):
        r"""
        We take a big picture, and chop it down to be processed by a NN in segments.
        Then we rearrange the image from its parts. 
        """
        self.model=model
        dir_3 = r"X:\Projects\Super_Rez\NN_Models"
        os.chdir(dir_3)
        if input_img.shape[2] < 4: #we have less than 4 channels
            mask_bool=False
        if model == "EDSR_x2":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "EDSR_x2.pb"
            sr.readModel(model_path)
            sr.setModel("edsr",2)
            if mask_bool:
                alpha_mask = self.transparency(input_img, tile_bool=False, factor=None)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "EDSR_x4":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "EDSR_x4.pb"
            sr.readModel(model_path)
            sr.setModel("edsr",4)
            if mask_bool:
                alpha_mask = self.transparency(input_img, tile_bool=False, factor=None)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "LapSRN_x8":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "LapSRN_x8.pb"
            sr.readModel(model_path)
            sr.setModel("lapsrn",8)
            if mask_bool:
                alpha_mask = self.transparency(input_img, tile_bool=False, factor=None)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "LapSRN_x4":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "LapSRN_x4.pb"
            sr.readModel(model_path)
            sr.setModel("lapsrn",4)
            if mask_bool:
                alpha_mask = self.transparency(input_img, tile_bool=False, factor=None)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "RealESRGAN_x4":
            sr=NN_Process.Runner()
            if mask_bool:
                alpha_mask = self.transparency(input_img, tile_bool=False, factor=None)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR) #we've made our alpha mask; NN can't handle transparency.
        final_image = sr.upsample(input_img)

        if mask_bool:
            b_channel, g_channel, r_channel = cv2.split(final_image)
            final_image = cv2.merge((b_channel, g_channel, r_channel, alpha_mask))
        return final_image
    #@function_timer
    def AI_resize_2(self, input_img, model="", factor=4, mask_bool=True):
        r"""
        We take a big picture, and chop it down to be processed by a NN in segments.
        Then we rearrange the image from its parts. 
        """
        self.model=model
        dir_3 = r"X:\Projects\Super_Rez\NN_Models"
        os.chdir(dir_3)
        if input_img.shape[2] < 4: #we have less than 4 channels
            mask_bool=False
        if model == "EDSR_x2":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "EDSR_x2.pb"
            sr.readModel(model_path)
            sr.setModel("edsr",2)
            if mask_bool:
                alpha_mask = self.transparency(input_img, factor=factor)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "EDSR_x4":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "EDSR_x4.pb"
            sr.readModel(model_path)
            sr.setModel("edsr",4)
            if mask_bool:
                alpha_mask = self.transparency(input_img, factor=factor)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "LapSRN_x8":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "LapSRN_x8.pb"
            sr.readModel(model_path)
            sr.setModel("lapsrn",8)
            if mask_bool:
                alpha_mask = self.transparency(input_img, factor=factor)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "LapSRN_x4":
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "LapSRN_x4.pb"
            sr.readModel(model_path)
            sr.setModel("lapsrn",4)
            if mask_bool:
                alpha_mask = self.transparency(input_img, factor=factor)
            # Set CUDA backend and target to enable GPU inference
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        elif model == "RealESRGAN_x4":
            sr=NN_Process.Runner()
            if mask_bool:
                alpha_mask = self.transparency(input_img, factor=factor)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR) #we've made our alpha mask; NN can't handle transparency.

        def image_divider(img):
            x_val, y_val = img.shape[1], img.shape[0]
            x_unit, y_unit = round(x_val/2), round(y_val/2)
            x_subunit, y_subunit = round(x_unit/4), round(y_unit/4)
            img_a, img_b = img[0:y_unit, 0:x_unit], img[0:y_unit, x_unit:x_unit*2]
            img_c, img_d = img[y_unit:y_unit*2, 0:x_unit], img[y_unit:y_unit*2, x_unit:x_unit*2]
            bar_v, bar_h = img[:,(x_unit-x_subunit):(x_unit+x_subunit)], img[(y_unit-y_subunit):(y_unit+y_subunit),:]
            return [img_a, img_b, img_c, img_d, bar_v, bar_h]
        
        def image_combiner(img_group):
            row_1 = np.concatenate((img_group[0], img_group[1]), axis=1)
            row_2 = np.concatenate((img_group[2], img_group[3]), axis=1)
            raw_result = np.concatenate((row_1, row_2), axis=0)
            raw_middle_v, raw_middle_h = round(raw_result.shape[1]/2), round(raw_result.shape[0]/2)

            #"""
            #apply seam correction; keep only the middle half of bars(edge inaccuracy)
            bar_v, bar_h = img_group[4], img_group[5]
            bar_v_fourth, bar_h_fourth = round(bar_v.shape[1]/4), round(bar_h.shape[0]/4)
            adj_bar_v, adj_bar_h = img_group[4][:,bar_v_fourth:bar_v_fourth*3], img_group[5][bar_h_fourth:bar_h_fourth*3,:]
            
            #could not broadcast input array from shape (1060,42,3) into shape (1096,42,3)
            raw_result[:,raw_middle_v-bar_v_fourth:raw_middle_v+bar_v_fourth] = adj_bar_v
            raw_result[raw_middle_h-bar_h_fourth:raw_middle_h+bar_h_fourth, :] = adj_bar_h
            #"""
            return raw_result

        def protocol(input_img):
            img_group = image_divider(input_img)
            for index, item in enumerate(img_group):
                img_group[index] = sr.upsample(item)

            final_image = image_combiner(img_group)       
            return final_image

        if factor == 4:
            final_image = protocol(input_img)

        elif factor == 16:
            print("NN Resizing - Factor:16", end="")
            img_group = image_divider(input_img)
            for index, item in enumerate(img_group):
                img_group[index] = protocol(item) # We're processing subquadrants rather than the whole image
            print("     COMPLETE - Combining Image")
            final_image = image_combiner(img_group)

        elif factor == 32:
            print("NN Resizing - Factor:32", end="")
            img_group = image_divider(input_img)

            for index, item in enumerate(img_group):
                item = image_divider(item)
                for subindex, subitem in enumerate(item):
                    item[subindex] = protocol(subitem) # We're processing sub-subquadrants rather than subquadrants
                img_group[index] = image_combiner(item)
            print("...............COMPLETE - Combining Image")
            final_image = image_combiner(img_group)

        else:
            print("Imminent Crash! Factors are powers of 2; e.g. 4, 16, 32, etc")

        if mask_bool:
            b_channel, g_channel, r_channel = cv2.split(final_image)
            final_image = cv2.merge((b_channel, g_channel, r_channel, alpha_mask))
        return final_image

    def kernel_sharpen(self, input_img=0, level=1.8):
        if level == 9:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        elif level == 5:
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        elif level == 3:
            kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])
        elif level == 1.8:
            kernel = np.array([[0,-0.2,0], [-0.2,1.8,-0.2], [0,-0.2,0]])
        img = cv2.filter2D(input_img, -1, kernel)
        return img

    # adjusts results to be even# length&width
    def percent_resize(self, input_img, percentage):
        scale_percent = percentage # percent of original size

        width = int(input_img.shape[1] * scale_percent / 100)
        height = int(input_img.shape[0] * scale_percent / 100)
        width = int(np.ceil(width /2) * 2)#Adjust to nearest even
        height = int(np.ceil(height /2) * 2)#Adjust to nearest even
        dim = (width, height)

        # resize image
        img = cv2.resize(input_img, dim, interpolation=cv2.INTER_LANCZOS4)
        #img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)
        #img = cv2.resize(input_img, dim, interpolation = cv2.INTER_LINEAR)
        #img = cv2.resize(input_img, dim, interpolation = cv2.INTER_CUBIC)
        return img
    
    def manual_resize(self, input_img, width, height):
        width = int(np.ceil(width /2) * 2)#Adjust to nearest even
        height = int(np.ceil(height /2) * 2)#Adjust to nearest even
        dim = (width, height)
        # resize image
        img = cv2.resize(input_img, dim, interpolation=cv2.INTER_LANCZOS4)
        #img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)
        #img = cv2.resize(input_img, dim, interpolation = cv2.INTER_LINEAR)
        return img

    def brightness_contrast(self, img, contrast=1, brightness=0):

        alpha = contrast # Contrast control (1.0-3.0)
        beta = brightness # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted
        #cv2.waitKey()

    def transparency(self, input_img, factor, tile_bool=True):
        r"""
        How transparency works:
        We take the source png, and copy its alpha channel into a dummy array that we enlarge(dumb)
        to the same size as the NN image. Then we apply the alpha mask to the output of the NN before we save.
        
        """
        #img = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA) # Convert to RGB with alpha channel
        alhpa_mask = np.ones((input_img.shape[0], input_img.shape[1]), dtype=input_img.dtype)
        for index, value in enumerate(input_img):
            for subindex, subvalue in enumerate(value):
                alhpa_mask[index][subindex] = subvalue[3]

        #alhpa_mask = self.manual_resize(alhpa_mask, x_val*size_multiple, y_val*size_multiple)

        #clone the alpha to 3 inputs needed for NN(3channel), then split the result to just alpha again.
        complex_alpha = cv2.merge((alhpa_mask, alhpa_mask, alhpa_mask))
        if factor != 4 and factor != None:
            print(f"Alpha Mask ", end="")
        
        if tile_bool:
            complex_alpha = self.AI_resize_2(complex_alpha, model=self.model, mask_bool=False, factor=factor)
        elif not tile_bool:
            complex_alpha = self.AI_resize_1(complex_alpha, model=self.model, mask_bool=False)
        
        alhpa_mask, m1, m2 = cv2.split(complex_alpha)
        return alhpa_mask

    def file_protocol(self, img):
        
        x_val, y_val = img.shape[1], img.shape[0]
        #maximum_multiple = np.sqrt(302500/(x_val*y_val)) #NN size limit 302,500px; n = srt(302,500/(x*y))
        #multiple_x, multiple_y = round(x_val*maximum_multiple), round(y_val*maximum_multiple)
        #img = resize.brightness_contrast(img, contrast=1.15)

        #Pass 1
        img = self.kernel_sharpen(img, level=1.8)
        model = "RealESRGAN_x4"
        if x_val <= 8 or y_val <= 8: #Too small
            pass
            #print(f"****************Too Tiny****************")
            #print(f"Skipped; under-sized: {file_name}, {x_val}x, {y_val}y; {source_dir}")
        
        elif x_val < 128 or y_val < 128:
            #print(f"Just a lil too small, better use basic NN: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = self.AI_resize_1(img, model="LapSRN_x8")

        elif x_val >= 4096 or y_val >= 4096: #Mega size
            #print(f"****************ENORMO COMIN THROUGH****************")
            #print(f"Huge Boy, 32x'er: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = self.AI_resize_2(img, model=model, factor=32)
        
        elif x_val >= 2048 or y_val >= 2048: #bigger than normal
            #print(f"We got a biggun, 16x I reckon: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = self.AI_resize_2(img, model=model, factor=16)
        
        elif x_val >= 1024 or y_val >= 1024: #too big for single pass, 4x needed.
            img = self.AI_resize_2(img, model=model, factor=4)
        
        else:#normal size
            img = self.AI_resize_1(img, model=model)

        cutoff_val = 1024
        if x_val>=cutoff_val and y_val>=cutoff_val: #big image
            img = self.manual_resize(img, x_val*2, y_val*2)
        else: #lil image
            img = self.manual_resize(img, x_val*4, y_val*4)

        return img

    def crawler(self, source_dir, load_extension="", save_extension=""):
        r"""
        Crawl over folder and subfolders, populate with new format and optional delete of old format.
        """
        if not os.path.isdir(self.out_dir): #Create output folder if it doesnt exist.
            os.mkdir(self.out_dir)

        os.chdir(source_dir)
        source_dir_list = os.listdir()

        for file in source_dir_list:
            os.chdir(source_dir)
            file_name = str(file)
            #print(f"{source_dir}  {file_name}")
            if file_name not in self.avoid_list:
                try:

                    # Check if file is folder; call recursion if true.
                    if os.path.isdir(file_name) and file_name not in self.forbidden_folders:
                        new_dir = source_dir+f"\{file_name}"
                        output_dir = self.out_dir+new_dir[len(self.parent_source_dir):]
                        if not os.path.isdir(output_dir): #Create output sub-folder if it doesnt exist.
                            os.mkdir(output_dir)
                        self.crawler(source_dir=new_dir, load_extension=load_extension, save_extension=save_extension)

                    # Not a folder, let's process the file if it matches our criteria.
                    elif not os.path.isdir(file_name) and file_name[-4:]==load_extension and file_name[-6:]!=self.forbidden_regex: #extensions match
                        os.chdir(source_dir)#In case we just finished a subfolder

                        # Read image
                        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
                        x_val, y_val = img.shape[1], img.shape[0]
                        if x_val > 8 and y_val > 8: #lets not save tiny imgs we cant process.
                            img = self.file_protocol(img) #this is where we determine how to AI-process the image.

                            output_dir = self.out_dir+source_dir[len(self.parent_source_dir):]
                            if not os.path.isdir(output_dir): #Create output sub-folder if it doesnt exist.
                                os.mkdir(output_dir)
                            os.chdir(output_dir)
                            save_name = file_name[0:-4]
                            cv2.imwrite(save_name+".png", img)

                        os.chdir(source_dir)
                        #os.remove(file_name) #delete the old png
                        self.i+=1
                        print(f"Cycle:{self.i}; {source_dir[len(self.parent_source_dir):]}...{file_name}")

                except Exception as e: 
                    #"""
                    if str(e)[94:113] != "(-217:Gpu API call)":
                        print(file_name, str(e), source_dir)
                    else:
                        print(f"{file_name} {x_val}x, {y_val}y ***Memory Error***  {source_dir}")
                    #"""
                    self.avoid_list.append(file_name)

def main(mode):
    if mode == "test":
        resize = NN_Resize()
        source_dir = r"X:\Projects\Super_Rez\Fallout NV\Source\Samples"
        output_dir = r"X:\Projects\Super_Rez\Fallout NV\Output"
        file_name = "outfitf"

        # Read image
        os.chdir(source_dir)
        img = cv2.imread(file_name+".png", cv2.IMREAD_UNCHANGED)
        x_val, y_val = img.shape[1], img.shape[0]
        maximum_multiple = np.sqrt(302500/(x_val*y_val)) #NN size limit 302,500px; n = srt(302,500/(x*y))
        multiple_x, multiple_y = round(x_val*maximum_multiple), round(y_val*maximum_multiple)

        #img = resize.kernel_sharpen(img, level=3)
        #img = resize.brightness_contrast(img, contrast=3, brightness=-10)
        #note:factor is how many many times we subdivide, for large images.
        model = "RealESRGAN_x4"
        
        img = resize.manual_resize(img, x_val*2, y_val*2)
        img = resize.kernel_sharpen(img, level=3)
        #img = resize.manual_resize(img, x_val, y_val)
        x_val, y_val = x_val*2, y_val*2

        if x_val <= 8 or y_val <= 8: #Too small
            print(f"****************Too Tiny****************")
            print(f"Skipped; under-sized: {file_name}, {x_val}x, {y_val}y; {source_dir}")
        
        elif x_val < 128 or y_val < 128:
            print(f"Just a lil too small, better use basic NN: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = resize.AI_resize_1(img, model="LapSRN_x8")

        elif x_val >= 4096 or y_val >= 4096: #Mega size
            print(f"****************ENORMO COMIN THROUGH****************")
            print(f"Huge Boy, 32x'er: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = resize.AI_resize_2(img, model=model, factor=32)
        
        elif x_val >= 2048 or y_val >= 2048: #bigger than normal
            print(f"We got a biggun, 16x I reckon: {file_name}, {x_val}x, {y_val}y; {source_dir}")
            img = resize.AI_resize_2(img, model=model, factor=16)
        
        elif x_val >= 1024 or y_val >= 1024: #too big for single pass, 4x needed.
            img = resize.AI_resize_2(img, model=model, factor=4)
        
        else:#normal size
            img = resize.AI_resize_1(img, model=model)

        #img = resize.brightness_contrast(img, contrast=.8, brightness=0)
        #img = resize.kernel_sharpen(img, level=3)
        #img = cv2.GaussianBlur(img,(3,3),0)
        #img = cv2.blur(img,(3,3))


        if x_val>=1000 and y_val>=1000: #big image
            img = resize.manual_resize(img, x_val*2, y_val*2)
        elif x_val<1000 and y_val<1000: #lil image
            img = resize.manual_resize(img, x_val*4, y_val*4)

        #img = resize.brightness_contrast(img, contrast=1.15, brightness=-5)
        #img = resize.kernel_sharpen(img, level=1.8)
        #img = cv2.GaussianBlur(img,(3,3),0)
        
        os.chdir(output_dir)
        cv2.imwrite(file_name+"_result.png", img)

    elif mode == "crawler":
        resize = NN_Resize()
        resize.parent_source_dir = r"X:\Projects\Super_Rez\Fallout NV\Source\DLCs"
        resize.out_dir = r"X:\Projects\Super_Rez\Fallout NV\Output\DLCs"
        source_dir = resize.parent_source_dir

        resize.crawler(source_dir, load_extension=".png", save_extension=".png")
        #quarentine_dir = r"X:\Projects\Super_Rez\Fallout NV\Source\Quarentine"

if __name__ == "__main__":
    main(mode="test")
