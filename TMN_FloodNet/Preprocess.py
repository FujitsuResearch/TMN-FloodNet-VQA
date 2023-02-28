import os
import json
import torch
import random
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import FloodNetVQA

def create_annotation_from_synthetic_data():
    '''
    Given Synthetic Datset, Create Compact Annotations as JSON file
    '''
    # Paths (Change Accordingly)
    base_path = r'C:\Users\akradiptad\OneDrive - FUJITSU\Desktop\Work\BIG - FRJ and MIT\FloodNet\Code\Data\Synthetic_Data'
    folder_list = os.listdir(base_path)

    id = 0

    data_list = []
    for folder in folder_list:
        # Collecting Each Scene
        sub_folder_list = os.listdir(os.path.join(base_path,folder))
        annotation_folder = sub_folder_list[0] if sub_folder_list[0].startswith('Dataset') else sub_folder_list[1]
        RGB_image_folder = sub_folder_list[1] if sub_folder_list[1].startswith('RGB') else sub_folder_list[0]
        
        # Mapping RGB with Sequence IDs
        f = open(os.path.join(base_path,folder,annotation_folder,'captures_000.json'))
        annotation_data = json.load(f)
        seq_id_rgb_dict = {}
        for entry in annotation_data['captures']:
            seq_id_rgb_dict[entry['sequence_id']] = entry['filename']

        # Annotation Mapping with RGBs
        annotation_files = ['metrics_000.json','metrics_001.json']
        for annotation_json in annotation_files:
            f = open(os.path.join(base_path,folder,annotation_folder,annotation_json))
            data = json.load(f)
            for entry in data['metrics']:
                if str(entry['metric_definition']).startswith('ObjectCount_2'):
                    data_row = {}
                    data_row['ID'] = id
                    data_row['Image_Filepath'] = str(os.path.join(base_path,folder,seq_id_rgb_dict[entry['sequence_id']])).replace('/','\\')
                    data_row['Objects'] = []
                    flood_condition = None
                    for obj_entry in entry['values']:
                        if obj_entry != None:
                            data_obj_row = {}
                            data_obj_row['Object_ID'] = obj_entry['label_id']
                            data_obj_row['Object_Name'] = obj_entry['label_name']
                            data_obj_row['Total_Count'] = obj_entry['count']
                            data_obj_row['Submerged_Count'] = obj_entry['count_submerged']
                            data_obj_row['Unaffected_Count'] = obj_entry['count_unaffected']
                            if data_obj_row['Object_Name'] == 'Road':
                                data_obj_row['Road_Flood_Condition'] = obj_entry['road_condition']
                                flood_condition = obj_entry['flood_condition']
                            data_row['Objects'].append(data_obj_row)
                        if obj_entry == None:
                            data_obj_row = {}
                            data_obj_row['Object_ID'] = 6
                            data_obj_row['Object_Name'] = 'Overall'
                            data_obj_row['Overall_Flood_Condition'] = flood_condition
                            data_row['Objects'].append(data_obj_row)
                    id += 1
                    data_list.append(data_row)

    json_object = json.dumps(data_list, indent=2)
    
    # Writing to sample.json
    with open(os.path.join(base_path, "Pretraining_Annotations.json"), "w") as outfile:
        outfile.write(json_object)


def create_questions_from_annotation():
    '''
    Given Synthetic Data Annotations, Create Questions and Labels for Dataloader
    '''
    # Paths
    base_path = r'C:\Users\akradiptad\OneDrive - FUJITSU\Desktop\Work\BIG - FRJ and MIT\FloodNet\Code\Data\Synthetic_Data'

    f = open(os.path.join(base_path,'Pretraining_Annotations.json'))
    data = json.load(f)

    image_list = []
    building_list = []
    road_list = []
    overall_list = []

    for row in data:
        image_list.append(row['Image_Filepath'])
        building_list.append([row['Objects'][3]['Total_Count'],row['Objects'][3]['Submerged_Count'],row['Objects'][3]['Unaffected_Count']])
        road_list.append([row['Objects'][4]['Total_Count'],row['Objects'][4]['Submerged_Count'],row['Objects'][4]['Unaffected_Count'],row['Objects'][4]["Road_Flood_Condition"]])
        overall_list.append([row['Objects'][5]['Overall_Flood_Condition']])

    overall_condition_questions = ["What is the overall condition of the given image?"]
    road_questions = ["What is the condition of road?", 
                    "What is the condition of the road in this image?"]
    road_yes_no_questions = ["Is the entire road flooded?",     
                            "Is the entire road non flooded?"]
    general_building_questions = ["How many buildings are in the image?", 
                                "How many buildings can be seen in the image?",
                                "How many buildings can be seen in this image?"]
    building_flooded_questions = ["How many non flooded buildings can be seen in this image?", 
                                "How many flooded buildings can be seen in this image?",
                                "How many buildings are non flooded?"]
    
    question_dict = {}
    id = 0

    for index in range(len(image_list)):
        
        # Overall condition
        entry_dict = {}
        entry_dict['Image_ID'] = image_list[index]
        entry_dict['Question'] = random.choice(overall_condition_questions)
        entry_dict['Ground_Truth'] = 'flooded' if overall_list[index] == 'Yes' else 'non flooded'
        entry_dict['Question_Type'] = "Condition_Recognition"
        question_dict[str(id)] = entry_dict
        id += 1

        # Road Condition
        entry_dict = {}
        entry_dict['Image_ID'] = image_list[index]
        entry_dict['Question'] = random.choice(road_questions)
        if road_list[index][-1] == 'Yes':
            entry_dict['Ground_Truth'] = 'flooded'
        elif road_list[index][-1] == 'No':
            entry_dict['Ground_Truth'] = 'non flooded'
        elif road_list[index][-1] == 'Partial':
            entry_dict['Ground_Truth'] =  'flooded,non flooded'
        entry_dict['Question_Type'] = "Condition_Recognition"
        question_dict[str(id)] = entry_dict
        id += 1

        # Road Condition (Yes-No)
        entry_dict = {}
        entry_dict['Image_ID'] = image_list[index]
        entry_dict['Question'] = random.choice(road_yes_no_questions)
        if entry_dict['Question'].find('non') != -1: # Non-Flooded Question
            if road_list[index][-1] == 'Yes' or road_list[index][-1] == 'Partial': # Road is Flooded or Partially Flooded
                entry_dict['Ground_Truth'] = 'No'
            elif road_list[index][-1] == 'No': # Road is not flooded
                entry_dict['Ground_Truth'] = 'Yes'
        elif entry_dict['Question'].find('non') == -1: # Flooded Question
            if road_list[index][-1] == 'Yes': # Road is Flooded
                entry_dict['Ground_Truth'] = 'Yes'
            elif road_list[index][-1] == 'No' or road_list[index][-1] == 'Partial': # Road is not flooded or Partially Flooded
                entry_dict['Ground_Truth'] = 'No'
        entry_dict['Question_Type'] = "Yes_No"
        question_dict[str(id)] = entry_dict
        id += 1

        # Simple Counting Question
        entry_dict = {}
        entry_dict['Image_ID'] = image_list[index]
        entry_dict['Question'] = random.choice(general_building_questions)
        entry_dict['Ground_Truth'] = int(building_list[index][0])
        entry_dict['Question_Type'] = "Simple_Counting"
        question_dict[str(id)] = entry_dict
        id += 1

        # Complex Counting Question
        entry_dict = {}
        entry_dict['Image_ID'] = image_list[index]
        entry_dict['Question'] = random.choice(building_flooded_questions)
        if entry_dict['Question'].find('non') != -1: # Non-flooded building questions 
            entry_dict['Ground_Truth'] = int(building_list[index][-1])
        elif entry_dict['Question'].find('non') == -1: # Flooded building questions 
            entry_dict['Ground_Truth'] = int(building_list[index][1])
        entry_dict['Question_Type'] = "Complex_Counting"
        question_dict[str(id)] = entry_dict
        id += 1

    json_object = json.dumps(question_dict, indent=2)
    
    # Writing to sample.json
    with open(os.path.join(base_path, "Synthetic_Train_Questions.json"), "w") as outfile:
        outfile.write(json_object)