
from torch.utils.data import DataLoader
import sys
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import torch.nn.functional as F
import nibabel as nib

def to1(img):
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img
def hu_300_to1(img):
    img[np.where(img <= -300)] = -300
    img[np.where(img >= 300)] = 300
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img
class Val_Dataset(dataset):
    def __init__(self, images_names):

        self.images_list = images_names

    def __getitem__(self, index):

        ct_array = nib.load(str(self.images_list[index])).get_fdata()

        seg_array = nib.load(str(self.images_list[index]).replace("images", "labels")).get_fdata()

        # ct_array = hu_300_to1(ct_array)
        # seg_array[seg_array < 10.] = 0.
        # seg_array[seg_array > 14.] = 0.
        # seg_array[seg_array == 11.] = 0.
        # seg_array[seg_array == 12.] = 0.
        # seg_array[seg_array == 13.] = 0.
        # seg_array[seg_array == 10.] = 1.
        # seg_array[seg_array == 14.] = 1.

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
        ct_array_resize = F.interpolate(ct_array, size=([128, 128, 160]), mode='trilinear', align_corners=False)
        ct_array_resize = ct_array_resize.squeeze(0).squeeze(0)
        ct_array_resize = ct_array_resize.numpy()

        seg_array = torch.FloatTensor(seg_array).unsqueeze(0).unsqueeze(0)
        seg_array_resize = F.interpolate(seg_array, size=([128, 128, 160]), mode="nearest")
        seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
        seg_array_resize = seg_array_resize.numpy()

        ct_array_resize = ct_array_resize.astype(np.float32)
        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)
        seg_array_resize = torch.FloatTensor(seg_array_resize)

        return ct_array_resize, seg_array_resize

    def __len__(self):
        return len(self.images_list)


    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

    def load_file_name_list_label(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.replace('_vol', '_seg').split())
        return file_name_list

class Val_Dataset_name_orgshape_nii(dataset):
    def __init__(self, images_names):

        self.images_list = images_names

    def __getitem__(self, index):

        ct_array = nib.load(str(self.images_list[index])).get_fdata()
        org_full_size = ct_array.shape
        seg_array = nib.load(str(self.images_list[index]).replace("images", "labels").replace("_0000.", ".")).get_fdata()

        # ct_array = hu_300_to1(ct_array)
        # seg_array[seg_array < 10.] = 0.
        # seg_array[seg_array > 14.] = 0.
        # seg_array[seg_array == 11.] = 0.
        # seg_array[seg_array == 12.] = 0.
        # seg_array[seg_array == 13.] = 0.
        # seg_array[seg_array == 10.] = 1.
        # seg_array[seg_array == 14.] = 1.
        
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
        ct_array_resize = F.interpolate(ct_array, size=([128, 128, 160]), mode='trilinear', align_corners=False)
        ct_array_resize = ct_array_resize.squeeze(0).squeeze(0)
        ct_array_resize = ct_array_resize.numpy()

        seg_array = torch.FloatTensor(seg_array).unsqueeze(0).unsqueeze(0)
        seg_array_resize = F.interpolate(seg_array, size=([128, 128, 160]), mode="nearest")
        seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
        seg_array_resize = seg_array_resize.numpy()

        ct_array_resize = ct_array_resize.astype(np.float32)
        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)
        seg_array_resize = torch.FloatTensor(seg_array_resize)

        return ct_array_resize, seg_array_resize, org_full_size, self.images_list[index]

    def __len__(self):
        return len(self.images_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

    def load_file_name_list_label(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.replace('_vol', '_seg').split())
        return file_name_list


class test_Dataset_name_orgshape_nii(dataset):
    def __init__(self, images_names):

        self.images_list = images_names

    def __getitem__(self, index):

        ct_array = nib.load(str(self.images_list[index])).get_fdata()
        org_full_size = ct_array.shape

        # ct_array = hu_300_to1(ct_array)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
        ct_array_resize = F.interpolate(ct_array, size=([256, 256, 128]), mode='trilinear', align_corners=False)
        ct_array_resize = ct_array_resize.squeeze(0).squeeze(0)
        ct_array_resize = ct_array_resize.numpy()

        ct_array_resize = ct_array_resize.astype(np.float32)
        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)

        return ct_array_resize, org_full_size, self.images_list[index]

    def __len__(self):
        return len(self.images_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

    def load_file_name_list_label(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.replace('_vol', '_seg').split())
        return file_name_list

class Val_Dataset_nii(dataset):
    def __init__(self, images_names):

        self.images_list = images_names
        self.labels_list = [n.replace('N4', 'v2_dskull_mask') for n in images_names]
        #
        # self.val_list = []
        # for root, dirs, files in os.walk(args.dataset_val_images_path):
        #     for file in files:
        #         self.val_list.append(file)
    def crop(self, image, seg):
        # 切割出预测结果部分，减少crf处理难度
        z = np.any(seg, axis=(1, 2))
        start_z, end_z = np.where(z)[0][[0, -1]]

        y = np.any(seg, axis=(0, 1))
        start_y, end_y = np.where(y)[0][[0, -1]]

        x = np.any(seg, axis=(0, 2))
        start_x, end_x = np.where(x)[0][[0, -1]]

        padding_size = 10
        # 扩张
        start_z = max(0, start_z - padding_size)
        start_x = max(0, start_x - padding_size)
        start_y = max(0, start_y - padding_size)

        end_z = min(image.shape[0], end_z + padding_size)
        end_x = min(image.shape[1], end_x + padding_size)
        end_y = min(image.shape[2], end_y + padding_size)

        image = image[start_z: end_z, start_x: end_x, start_y: end_y]
        seg = seg[start_z: end_z, start_x: end_x, start_y: end_y]

        return image, seg

    def __getitem__(self, index):

        ct_array = nib.load(str(self.images_list[index])).get_fdata()
        seg_array = nib.load(str(self.labels_list[index])).get_fdata()
        ct_array = to1(ct_array)
        seg_array[seg_array != 0.] = 1.

        # ct_array, seg_array = self.crop(ct_array, seg_array)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
        ct_array_resize = F.interpolate(ct_array, size=([160, 160, 160]), mode='trilinear', align_corners=False)
        ct_array_resize = ct_array_resize.squeeze(0).squeeze(0)
        ct_array_resize = ct_array_resize.numpy()

        seg_array = torch.FloatTensor(seg_array).unsqueeze(0).unsqueeze(0)
        seg_array_resize = F.interpolate(seg_array, size=([160, 160, 160]), mode="nearest")
        seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
        seg_array_resize = seg_array_resize.numpy()
        # ct_array_resize = ct_array
        # seg_array_resize = seg_array
        # seg_array_resize = seg_array_resize.astype(np.uint8)
        # '''
        # 下面对label进行多通道的处理
        # '''

        # ct_array_resize = ct_array_resize / self.args.norm_factor
        ct_array_resize = ct_array_resize.astype(np.float32)

        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)
        seg_array_resize = torch.FloatTensor(seg_array_resize)

        # if self.transforms:
        #     ct_array,seg_array = self.transforms(ct_array, seg_array)
        # this_name = self.images_list[index].split('\\')[-1]


        return ct_array_resize, seg_array_resize, self.images_list[index]    # torch.Size([2, 1, 128, 128, 128]) torch.Size([2, 128, 128, 128])

    def __len__(self):
        return len(self.images_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

    def load_file_name_list_label(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.replace('_vol', '_seg').split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    val_ds = Val_Dataset(args)

    # 定义数据加载
    val_dl = DataLoader(val_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(val_dl):
        print(i,ct.size(),seg.size())