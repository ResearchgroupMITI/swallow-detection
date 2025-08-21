import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
import re
import torch.nn.functional as F
from torchvision import transforms

class SwallowDataset:
    def __init__(self, hparams, idx, val_splits):
        self.name = "Swallow"
        self.hparams = hparams
        self.idx = idx
        self.val_splits = val_splits
        self.data = self.create_data_sets()

    def create_data_sets(self):
        dataset = {}
        for split in ["train", "val"]:
            dataset[split] = Swallow_Helper(self.hparams, split, self.idx, self.val_splits)
        return dataset

    @staticmethod
    def add_dataset_specific_args(parser):
        data_args = parser.add_argument_group(title='data args options')
        data_args.add_argument("--resize_type", type=str, default="224x224")
        data_args.add_argument("--num_classes", type=int, default=2)
        data_args.add_argument("--num_patients", type=int, default=5)
        return parser

class Swallow_Helper(Dataset):
    def __init__(self, hparams, mode, idx, val_splits):
        self.hparams = hparams
        self.img_dir = os.path.join(self.hparams.data_root, 'tensors')
        self.anno_dir = os.path.join(self.hparams.data_root, 'anno')
        self.pregenerated_list_path = os.path.join(self.hparams.data_root, "pregenerated")
        self.transformations = self.get_transformations()
        self.class_names = ['notswallow', 'swallow']
        self.class_name_to_num = {}
        self.mode = mode
        self.idx = idx
        self.val_splits = val_splits

        for id, feature_name in enumerate(self.class_names):
            self.class_name_to_num[feature_name] = id

        self.img_list, self.anno_list = self.get_img_list()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_pt = torch.load(img_path)
        img_pt = torch.stack([img_pt] * 3, dim=0).to(torch.float)
        label = self.anno_list[idx]
        id = label['img_nr']
        label_noid = label.drop(['img_nr', 'label'])
        label_test = label_noid.values
        label_test = label_test.astype(float)
        label_tensor = torch.Tensor(label_test)

        return img_pt, label_tensor, id

    def get_transformations(self):
        """tbd"""
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
  
        if self.hparams.resize_type == "224x224":
            resize_height = 224
            resize_width = 224
        elif self.hparams.resize_type == "448x448":
            resize_height = 448
            resize_width = 448

        data_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(resize_height, resize_width)),
            transforms.ToTensor(),
            normalize
        ])
        return data_transformations

    def get_img_list(self):
        """Returns lists containig image paths and corresponding annoations"""
        # Create directory for pregenerated img list, if not already exists
        os.makedirs(self.pregenerated_list_path, exist_ok=True)
        img_anno_list_path = os.path.join(self.pregenerated_list_path, f"fold{self.idx+1}_{self.mode}.pickle")

        # Check if the pickled lists already exist
        if os.path.isfile(img_anno_list_path):
            # Load the existing lists
            with open(img_anno_list_path, "rb") as fp:
                img, anno = pickle.load(fp)
        else:
            # Generate lists
            img_list, anno_list = self.generate_imganno_lists()

            img, anno= self.split_folds(img_list, anno_list)
            list_path =  os.path.join(self.pregenerated_list_path, f"fold{self.idx+1}_{self.mode}.pickle")

            # Save the image and annotation lists to a file
            with open(list_path, "wb") as fp:
                print(f"Saving fold{self.idx+1} to {list_path}")
                pickle.dump([img, anno], fp)

        return img, anno

    def generate_imganno_lists(self):
        """Generates lists containing the image paths and annotations"""
        # Initialize empty image and annotation lists
        img_list = []
        anno_list = []

        print(f"Creating fold{self.idx+1}_{self.mode}")

        # Iterate through the directories in img_dir
        for patient in os.listdir(self.img_dir):

            # Read the corresponding annotation files
            anno_path = os.path.join(self.anno_dir, f"{patient}.csv" )
            anno = pd.read_csv(anno_path, sep=",")

            print(f"Loading {patient} with {anno.shape[0]} frames")

            # Iterate through the annotation rows
            for _, anno_row in anno.iterrows():
                img_tensor_path = os.path.join(self.img_dir, patient, f"{str(anno_row.img_nr)}.pt")

                # Check if the image file exists (sometimes the annotations are a bit longer than the video)
                if os.path.isfile(img_tensor_path):
                    assert 'img_nr' in anno_row
                    img_list.append(img_tensor_path)

                    feature = torch.tensor(anno_row['label'])
                    feature_1h_pd = F.one_hot(feature, num_classes=self.hparams.num_classes)
                    feature_1h = pd.DataFrame(feature_1h_pd, index=[cl for cl in self.class_names])

                    anno_row_ = pd.concat([anno_row, feature_1h]).squeeze()

                    anno_list.append(anno_row_)
                else:
                    print(f"No frame for annotation {anno_row.id}")
        return img_list, anno_list

    def split_folds(self, img_list, anno_list):

        # Split data based on patient IDs
        img_train = []
        anno_train = []
        img_val = []
        anno_val= []

        for idx, img_path in enumerate(img_list):
            #patient_id = os.path.basename(os.path.dirname(img_path))
            patient_id = int(re.search(r'patient_(\d+)', img_path).group(1))
            if patient_id in self.val_splits[self.idx]:
            # if patient_id == patient:  # Move patient1 to validation set
                img_val.append(img_path)
                anno_val.append(anno_list[idx])
            else:
                img_train.append(img_path)
                anno_train.append(anno_list[idx])

        if self.mode == "train":
            return img_train, anno_train
        if self.mode == "val":
            return img_val, anno_val

