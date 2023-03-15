import os
import json
import torch
import numpy as np
import transformers
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Pretraining_Count_Data_FloodNetVQA(Dataset):
    def __init__(self, dataroot, partition, height, width):
        self.partition = partition
        self.im_height = height
        self.im_width = width
        self.base_path = dataroot
        self.preprocess_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.Resize((self.im_height,self.im_width)),
                                                    transforms.ToTensor()])
        self.preprocess_val_test = transforms.Compose([transforms.Resize((self.im_height,self.im_width)),
                                                       transforms.ToTensor()])
        
        self.image_list = []
        self.vehicle_list = []
        self.tree_list = []
        self.person_list = []
        self.building_list = []
        self.road_list = []

        f = open(os.path.join(self.base_path,'Pretraining_Annotations.json'))
        self.data = json.load(f)

        for row in self.data:
            self.image_list.append(row['Image_Filepath'])
            self.vehicle_list.append([row['Objects'][0]['Total_Count'],row['Objects'][0]['Submerged_Count'],row['Objects'][0]['Unaffected_Count']])
            self.tree_list.append([row['Objects'][1]['Total_Count'],row['Objects'][1]['Submerged_Count'],row['Objects'][1]['Unaffected_Count']])
            self.person_list.append([row['Objects'][2]['Total_Count'],row['Objects'][2]['Submerged_Count'],row['Objects'][2]['Unaffected_Count']])
            self.building_list.append([row['Objects'][3]['Total_Count'],row['Objects'][3]['Submerged_Count'],row['Objects'][3]['Unaffected_Count']])
            self.road_list.append([row['Objects'][4]['Total_Count'],row['Objects'][4]['Submerged_Count'],row['Objects'][4]['Unaffected_Count']])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # Image
        image = Image.open(self.image_list[index]).convert('RGB')
        
        if self.partition == 'train':
            image = self.preprocess_train(image).type(torch.FloatTensor)
        else:
            image = self.preprocess_val_test(image).type(torch.FloatTensor)
        
        # Count Labels
        vehicle_count_total = self.vehicle_list[index][0]
        vehicle_count_submerged = self.vehicle_list[index][1]
        vehicle_count_unaffected = self.vehicle_list[index][2]

        tree_count_total = self.tree_list[index][0]
        tree_count_submerged = self.tree_list[index][1]
        tree_count_unaffected = self.tree_list[index][2]

        person_count_total = self.person_list[index][0]
        person_count_submerged = self.person_list[index][1]
        person_count_unaffected = self.person_list[index][2]

        building_count_total = self.building_list[index][0]
        building_count_submerged = self.building_list[index][1]
        building_count_unaffected = self.building_list[index][2]

        road_count_total = self.road_list[index][0]
        road_count_submerged = self.road_list[index][1]
        road_count_unaffected = self.road_list[index][2]

        return [image,
                vehicle_count_total,vehicle_count_submerged,vehicle_count_unaffected,
                tree_count_total,tree_count_submerged,tree_count_unaffected,
                person_count_total,person_count_submerged,person_count_unaffected,
                building_count_total,building_count_submerged,building_count_unaffected,
                road_count_total,road_count_submerged,road_count_unaffected]