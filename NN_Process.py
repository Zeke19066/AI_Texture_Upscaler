import cv2
import math
import numpy as np
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.hub import download_url_to_file, get_dir
from torch.nn import functional as F
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():

    def __init__(self, scale, model_path, tile_pad=10, pre_pad=10, half=False):
        self.scale = scale
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)


        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            #Padding size should be less than the corresponding input dimension, but got: padding (0, 10) at dimension 3 of input [1, 3, 512, 6]
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')


    def process(self):
        self.output = self.model(self.img)

    def post_process(self):

        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]

        return self.output

    @torch.no_grad()
    def enhance(self, img):
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 255:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) ####################
        return output

class Runner():
    def __init__(self):
        pass
    def upsample(self, img):
        """
        CUDA out of memory. 
        Tried to allocate 4.08 GiB 
        (GPU 0; 8.00 GiB total capacity; 2.37 GiB already allocated; 3.79 GiB free; 2.50 GiB reserved in total by PyTorch) 
        """
        model_path = r"X:\Projects\Super_Rez\NN_Models\RealESRGAN_x4plus.pth"
        x_val, y_val = img.shape[1], img.shape[0]
        scale = 2
        resize = RealESRGANer(scale, model_path, pre_pad=0, half=False)
        img = resize.enhance(img)
        
        dim = (int(x_val*4), int(y_val*4))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
        return img

def main():
    source_dir = r"X:\Projects\Super_Rez\Fallout NV\Source\Samples_HL"
    output_dir = r"X:\Projects\Super_Rez\Fallout NV\Output"
    file_name = "nv_signs_state"

    # Read image
    os.chdir(source_dir)
    img = cv2.imread(file_name+".png", cv2.IMREAD_UNCHANGED)
    sr=Runner()
    img=sr.upsample(img)
    os.chdir(output_dir)
    cv2.imwrite(file_name+"_result.png", img)

if __name__ == "__main__":
    main()