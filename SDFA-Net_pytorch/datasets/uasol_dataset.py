import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as tf

from datasets.utils.data_reader import get_input_img, get_camera_params
from datasets.utils.my_transforms import BatchRandomCrop, NoneTransform
from path_my import Path
from utils import platform_manager

K_of_KITTI = [[721.54, 0, 609.56, 0],
              [0, 721.54, 172.85, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

@platform_manager.DATASETS.add_module
class UASOLDataset(data.Dataset):

    def __init__(self,
                 dataset_mode,
                 split_file,
                 crop_coords=[0, 0, 768, 2048],
                 full_size=None,
                 patch_size=None,
                 normalize_params=[0.411, 0.432, 0.45],
                 load_KTmatrix=False,
                 flip_mode=None, # "img", "k", "both" (lr, ud, rotation)
                 ):
        super().__init__()

        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('uasol')
        self.split_file = split_file
        self.full_size = full_size
        self.patch_size = patch_size
        self.flip_mode = flip_mode

        self.crop_coords = crop_coords

        self.KTmatrix = load_KTmatrix

        self.file_list = self._get_file_list(split_file)


        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])
        if dataset_mode == 'train':
            # random resize and crop
            if self.patch_size is not None:
                self.crop = BatchRandomCrop(patch_size)
            else:
                self.crop = NoneTransform()
        else:
            if self.full_size is not None:
                self.color_resize = tf.Resize(full_size,
                                              interpolation=Image.ANTIALIAS)
            else:
                self.color_resize = NoneTransform()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, f_idx):
        """Return a data dictionary."""
        file_info = self.file_list[f_idx]
        color_l_path = os.path.join(self.dataset_dir, file_info["color_frame_left"])

        # Read data
        inputs = {}
        inputs['color_s_raw'] = get_input_img(color_l_path + ".png")

        inputs['direct'] = torch.tensor(1, dtype=torch.float)

        if self.dataset_mode == "train":
            color_r_path = os.path.join(self.dataset_dir, file_info["color_frame_right"])
            inputs['color_o_raw'] = get_input_img(color_r_path + ".png")

        depth_path = os.path.join(self.dataset_dir, file_info["depth_frame"])
        inputs['depth'] = get_input_img(depth_path + ".png")

        intrinsic = np.array(K_of_KITTI)
        k = intrinsic[0, 0] * 0.54
        inputs['disp_k'] = torch.tensor(k, dtype=torch.float)
        if self.full_size is not None:
            inputs['disp_k'] *= self.full_size[1] / 1244

        # Process data
        # resize crop & flip for train
        if self.dataset_mode == 'train':
            # crop for image
            if self.crop_coords is not None:
                self.fix_crop = tf.functional.crop
            else:
                self.fix_crop = NoneTransform()

            # resize
            if self.full_size is not None:
                img_size = self.full_size
            else:
                _size = inputs["color_s_raw"].size # (w, h)
                img_size = (_size[1], _size[0])
            scale_factor = random.uniform(0.75, 1.5)\
                if self.patch_size is not None else 1
            if scale_factor != 1 or self.full_size is not None:
                random_size = tuple(int(s * scale_factor) for s in img_size)
                self.color_resize = tf.Resize(random_size,
                                              interpolation=Image.ANTIALIAS)
                self.depth_resize = tf.Resize(random_size,
                                              interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.depth_resize = NoneTransform()
            # crop
            crop_params = self.crop.get_params(img_size, self.patch_size,
                                               scale_factor)

            # flip
            is_flip = (self.dataset_mode == 'train' and
                       self.flip_mode is not None and
                       random.uniform(0, 1) > 0.5)
            if is_flip:
                if self.flip_mode == "both": # random flip mode
                    flip_img = random.uniform(0, 1) > 0.5
                else:
                    flip_img = self.flip_mode == "img"

                if flip_img:
                    inputs['color_o_raw'], inputs['color_s_raw'] =\
                        inputs['color_s_raw'], inputs['color_o_raw']
                else:  # flip_mode == "k"
                    inputs['direct'] = -inputs['direct']

            for key in list(inputs):
                if 'color' in key:
                    raw_img = inputs[key]
                    if is_flip:
                        raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                    img = self.crop(self.color_resize(raw_img), crop_params)
                    img = self.to_tensor(img)
                    inputs[key.replace("_raw", "")] =\
                        self.normalize(img)
                    inputs[key.replace("_raw", "_aug")] =\
                        self.normalize(img)
                elif 'depth' in key:
                    # depth will be changed when resize
                    raw_depth = np.array(inputs[key]) / scale_factor
                    if is_flip:
                        raw_depth = np.fliplr(raw_depth)
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)[:, :, 0]
                    depth = self.crop(self.depth_resize(depth), crop_params)
                    inputs[key] = depth
        
        else:
            for key in list(inputs):
                if 'color' in key:
                    raw_img = inputs[key]
                    img = self.color_resize(raw_img)
                    inputs[key.replace('_raw', '')] =\
                        self.normalize(self.to_tensor(img))
                elif 'depth' in key:
                    # do not resize ground truth in test
                    raw_depth = np.array(inputs[key])[:, :, 0]
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)  
                    inputs[key] = depth
                    
        # delete raw data
        inputs.pop("color_s_raw")
        if self.dataset_mode == 'train':
            inputs.pop("color_o_raw")      
        
        inputs["file_info"] = [file_info]

        return inputs


    def _get_file_list(self, split_file):
        f = open(split_file, 'r')
        files = json.load(f)
        return files["Data"]

    @property
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        return infos

        

