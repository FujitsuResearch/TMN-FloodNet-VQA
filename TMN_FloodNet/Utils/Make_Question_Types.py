import os
import json

qdict = json.load(open("/DATA/FloodNet/Code/Data/Questions/Test_Questions_Programs.json", "r"))

def do(jsondict):
    qs = {}
    for key, example in jsondict.items():
        if example["Question"] == "What is the condition of the road in this image?" or example["Question"] == "What is the condition of road?":
            qs[key] = example
            qs[key]["Question_Type"] = "Road_Condition_Recognition"
        else:
            qs[key] = example
    
    root_path = "/DATA/FloodNet/Code/Data/Questions/"
    with open(os.path.join(root_path, "New_Train_Questions_Programs.json"), "w") as outfile:
        json.dump(qs, outfile)

do(qdict)