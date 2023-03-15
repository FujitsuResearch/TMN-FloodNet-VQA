import os
import json
from sklearn.model_selection import train_test_split

qdict = json.load(open("/DATA/FloodNet/Code/Data/Questions/Question_Programs.json", "r"))

def get_uniq_image_ids(jsondict):
    uniq_images = []
    for key, example in jsondict.items():
        if example["Image_ID"] not in uniq_images:
            uniq_images.append(example["Image_ID"])
    return uniq_images

def get_entry_from_imageid_list(jsondict, imageid_list):
    qs = {}
    for imageid in imageid_list:
        for key, example in jsondict.items():
            if example["Image_ID"] == imageid:
                qs[key] = example
    return qs

def get_train_test_splits_imagewise(jsondict):
    root_path = "/DATA/FloodNet/Code/Data/Questions/"
    train_dict = {}
    test_dict = {}
    uniq_images = get_uniq_image_ids(jsondict)
    train, test = train_test_split(uniq_images, test_size=0.2, random_state = 42)
    train_dict = get_entry_from_imageid_list(jsondict,train)
    test_dict = get_entry_from_imageid_list(jsondict,test) 
    
    train_keys = list(train_dict.keys()) 
    test_keys = list(test_dict.keys())

    flag = False
    for i in train_keys:
        if i in test_keys:
            flag = True

    # Check if Train and Test Contains Disjoint Data
    if flag == False:
        print('Success: Training and Testing Files contain disjoint data')
    else:
        print('Error: Training and Testing Files DOES NOT contain disjoint data')

    print('No of Training Questions: ', len(train_keys))
    print('No of Testing Questions: ', len(test_keys))
    
    with open(os.path.join(root_path, "Train_Questions_Programs.json"), "w") as outfile:
        json.dump(train_dict, outfile)
    with open(os.path.join(root_path, "Test_Questions_Programs.json"), "w") as outfile:
        json.dump(test_dict, outfile)
    
get_train_test_splits_imagewise(qdict)