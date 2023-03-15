import os
import sys
sys.path.append("../../FloodNet/Code")

class PATH:
    def __init__(self):

        # self.root_path = os.getcwd()
        # self.dataset_path = os.path.join(self.root_path, "Data")
        # self.results_path = os.path.join(self.root_path,"TMN_FloodNet","Results")
        # self.storage_path = os.path.join(self.results_path,"FloodNet_VT")
        
        # # Model Paths
        # self.save_path = os.path.join(self.storage_path,'Model')
        # self.config_file = os.path.join(self.root_path,"TMN","config","bert_base_6layer_6conect.json")
        # self.bert_model = os.path.join(self.root_path,"TMN","config","bert-base-uncased-vocab.txt")
        
        # # Dataset Paths
        # self.vocab_path = os.path.join(self.root_path,"Data","Vocabs","Answers_Vocab.json")
        # self.func_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Functions_Vocab.json")
        # self.args_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Arguments_Vocab.json")

        
        self.root_path = "/groups/gcd50678/datasets/floodnet"
        self.dataset_path = os.path.join(self.root_path, "Data")
        # self.results_path = os.path.join(self.root_path,"TMN_FloodNet","Results")
        # self.storage_path = os.path.join(self.results_path,"FloodNet_VT")
        # self.storage_path = "/groups/gcd50678/results/TMN_FloodNet"
        
        # Model Paths
        # self.save_path = os.path.join(self.storage_path,'Model')
        self.save_path = "/groups/gcd50678/results/TMN_FloodNet"
        self.config_path = "/home/acb11247il/dev/TMN-FloodNet-VQA/"
        self.config_file = os.path.join(self.config_path,"TMN","config","bert_base_6layer_6conect.json")
        self.bert_model = os.path.join(self.config_path,"TMN","config","bert-base-uncased-vocab.txt")
        
        # Dataset Paths
        self.vocab_path = os.path.join(self.root_path,"Data","Vocabs","Answers_Vocab.json")
        self.func_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Functions_Vocab.json")
        self.args_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Arguments_Vocab.json")

        self.csv_path = os.path.join(self.config_path, "results.csv")