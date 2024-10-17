import cv2
import os
import glob
import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, cover_dir, select_num,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cover_dir = cover_dir
        self.select_num = select_num        
        self.cover_list = [x.split('/')[-1] for x in glob.glob(cover_dir + '/*')]
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.select_num)

    def __getitem__(self, idx):
        idx = int(idx)
        idx = self.select_num[idx]
        cover_path = os.path.join(self.cover_dir, 
                                  self.cover_list[idx])

        data = cv2.imread(cover_path, -1)
        sample = {'data': data}

        if self.transform:
            sample = self.transform(sample)

        return sample
