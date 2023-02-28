import os

class PATH:
    def __init__(self):

        self.root_path = r"C:\Users\akradiptad\OneDrive - FUJITSU\Desktop\Work\BIG - FRJ and MIT\FloodNet\Code"
        self.dataset_path = os.path.join(self.root_path, "Data")
        self.results_path = os.path.join(self.root_path,"TMN_FloodNet","Results")
        self.storage_path = os.path.join(self.results_path,"FloodNet_VT")
        
        # Model Paths
        self.save_path = os.path.join(self.storage_path,'Model')
        self.config_file = os.path.join(self.root_path,"TMN","config","bert_base_6layer_6conect.json")
        self.bert_model = os.path.join(self.root_path,"TMN","config","bert-base-uncased-vocab.txt")
        
        # Dataset Paths
        self.vocab_path = os.path.join(self.root_path,"Data","Vocabs","Answers_Vocab.json")
        self.func_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Functions_Vocab.json")
        self.args_vocab_path = os.path.join(self.root_path,"Data","Vocabs","Arguments_Vocab.json")