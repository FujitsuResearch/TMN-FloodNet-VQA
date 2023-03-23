import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split

qdict = json.load(open("/DATA/FloodNet/Code/Data/Questions/Test_Questions_Programs.json", "r"))

def get_uniq_ques_type_cnt(jsondict):
    freq_dict = {'Condition_Recognition':0,
                 'Road_Condition_Recognition':0,
                 'Simple_Counting':0, 
                 'Complex_Counting':0, 
                 'Yes_No':0}
    for key, example in jsondict.items():
        if example["Question_Type"] == 'Condition_Recognition':
            freq_dict['Condition_Recognition'] += 1
        elif example["Question_Type"] == 'Road_Condition_Recognition':
            freq_dict['Road_Condition_Recognition'] += 1
        elif example["Question_Type"] == 'Simple_Counting':
            freq_dict['Simple_Counting'] += 1
        elif example["Question_Type"] == 'Complex_Counting':
            freq_dict['Complex_Counting'] += 1
        elif example["Question_Type"] == 'Yes_No':
            freq_dict['Yes_No'] += 1
    print(freq_dict)
    return freq_dict

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

def get_train_test_splits_queswise(jsondict):
    train_dict = {}
    test_dict = {}
    id_dict = {'Condition_Recognition':[],
               'Road_Condition_Recognition':[],
               'Simple_Counting':[], 
               'Complex_Counting':[], 
               'Yes_No':[]}
    for key, example in jsondict.items():
        if example["Question_Type"] == 'Condition_Recognition':
            id_dict['Condition_Recognition'].append(int(key))
        elif example["Question_Type"] == 'Road_Condition_Recognition':
            id_dict['Road_Condition_Recognition'].append(int(key))
        elif example["Question_Type"] == 'Simple_Counting':
            id_dict['Simple_Counting'].append(int(key))
        elif example["Question_Type"] == 'Complex_Counting':
            id_dict['Complex_Counting'].append(int(key))
        elif example["Question_Type"] == 'Yes_No':
            id_dict['Yes_No'].append(int(key))
    
    for key in id_dict:
        train_ids, test_ids = train_test_split(id_dict[key], )
    
def do(jsondict):
    freq_dict = {'Condition_Recognition':0,
                 'Road_Condition_Recognition':0,
                 'Simple_Counting':0, 
                 'Complex_Counting':0, 
                 'Yes_No':0}
    dd = {'Yes':0, 'No':0}
    for key, example in jsondict.items():
        if example["Question_Type"] == 'Yes_No':
            if example["Ground_Truth"] == 'Yes':
                dd['Yes'] += 1
            elif example["Ground_Truth"] == 'No':
                dd['No'] += 1
    print(dd)

def do2(jsondict):
    freq_dict = {'Condition_Recognition':0,
                 'Road_Condition_Recognition':0,
                 'Simple_Counting':0, 
                 'Complex_Counting':0, 
                 'Yes_No':0}
    dd = []
    for key, example in jsondict.items():
        if example["Question_Type"] == 'Simple_Counting':
            dd.append(example["Ground_Truth"])
    dd_count = dict(Counter(dd))
    print(dd_count)

def do3(jsondict):
    freq_dict = {'Condition_Recognition':0,
                 'Road_Condition_Recognition':0,
                 'Simple_Counting':0, 
                 'Complex_Counting':0, 
                 'Yes_No':0}
    dd = []
    for key, example in jsondict.items():
        if example["Question_Type"] == 'Complex_Counting':
            dd.append(example["Ground_Truth"])
    dd_count = dict(Counter(dd))
    print(dd_count)

#get_train_test_splits_queswise(qdict)
#get_train_test_splits_imagewise(qdict)
#get_uniq_ques_type_cnt(qdict)
do(qdict)
do2(qdict)
do3(qdict)