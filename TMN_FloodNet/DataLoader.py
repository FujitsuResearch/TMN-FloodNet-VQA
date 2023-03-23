import os
import json
import torch
import numpy as np
import transformers
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Synthetic_Dataset(Dataset):
    def __init__(self, dataroot, height, width):
        """Dataloader for Synthetically Generated VQA dataset
           Inputs:
                dataroot (str): Path to Dataset (Image and Questions)
                partition (str): Partition Specifier ('Train', 'Val', 'Test')
                height (int): Image Height
                width (int): Image Width 
           Outputs:
                Image (tensor): Image Tensor [Batchsize x Channels x Height x Width]
                question (tuple): Tuple of Question strings in the batch
                question_type (tuple): Tuple of Question types for the VQA dataset
                Program String (tuple): Tuple of Program strings in the batch
                args (tensor): Encoded program [Batchsize x 2(Max_Program_Length) x 3(Max_Arguments + 1 for Function)]
                answer (tensor): Encoded Answers/Ground Truth Labels [Batchsize x 1]
        """
        self.syntheic_dataroot = "/DATA/FloodNet/Code/Data/Synthetic_Data"
        self.im_height = height
        self.im_width = width
        self.args_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Arguments_Vocab.json'), 'r'))
        self.func_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Functions_Vocab.json'), 'r'))
        self.ans_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Answers_Vocab.json'), 'r'))
        
        self.preprocess_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.Resize((self.im_height,self.im_width)),
                                                    transforms.ToTensor()])
        self.preprocess_val_test = transforms.Compose([transforms.Resize((self.im_height,self.im_width)),
                                                       transforms.ToTensor()])
        self.question_list = []
        self.question_type_list = []
        self.program_list = []
        self.image_list = []
        self.answer_list = []

        question_file = open(os.path.join(self.syntheic_dataroot,'Synthetic_Train_Questions_Programs.json'))
        data = json.load(question_file)

        for key in list(data.keys()):
            q = data[key]['Question']
            q_type = data[key]['Question_Type']
            program = data[key]['Program']
            img_file = data[key]['Image_ID']
            ans = data[key]['Ground_Truth']
            self.question_list.append(q)
            self.question_type_list.append(q_type)
            self.program_list.append(program)
            self.image_list.append(img_file)
            self.answer_list.append(ans)

    def __len__(self):
        return len(self.image_list)

    def _prog_string_to_list(self, prog_string):
        """Returns formatted list version of Program String for Encoding
        """
        prog_string = prog_string.split(' & ')
        prog_list = []
        for i in prog_string:
            dict = {}
            prog = 'None'
            vals = 'None'
            if '(' in i:
                prog = i[:i.index('(')]
                vals = i[i.index('(')+1:i.index(')')].split(', ')
            else:
                prog = i
            dict['function'] = prog
            dict['argument'] = vals
            prog_list.append(dict)
        return prog_list

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Image
        image = Image.open(self.image_list[index]).convert('RGB')
        image = self.preprocess_train(image).type(torch.FloatTensor)
        
        # Question
        question = self.question_list[index]
        # Question Type
        qs_type = self.question_type_list[index]
        
        # Program
        program_string = self.program_list[index]
        program = self._prog_string_to_list(self.program_list[index])
        args = np.zeros((2, 3))
        num_progs = 0
        for p in program:
            func = p['function']
            arg_list = p['argument']
            if arg_list != 'None':
                for i in range(len(arg_list)):
                    arg_idx = self.args_vocab[arg_list[i]]
                    args[num_progs][i+1] = arg_idx
            func_idx = self.func_vocab[func]
            args[num_progs][0] = func_idx
            num_progs = num_progs + 1
        args = torch.from_numpy(args).type(torch.FloatTensor)
        # Answer
        answer = self.answer_list[index]
        answer = self.ans_vocab[str(answer)]
        return image, question, qs_type, program_string, args, answer

class FloodNetVQA(Dataset):
    def __init__(self, dataroot, partition, height, width):
        """Dataloader for FloodNet VQA dataset
           Inputs:
                dataroot (str): Path to Dataset (Image and Questions)
                partition (str): Partition Specifier ('Train', 'Val', 'Test')
                height (int): Image Height
                width (int): Image Width 
           Outputs:
                Image (tensor): Image Tensor [Batchsize x Channels x Height x Width]
                question (tuple): Tuple of Question strings in the batch
                question_type (tuple): Tuple of Question types for the VQA dataset
                Program String (tuple): Tuple of Program strings in the batch
                args (tensor): Encoded program [Batchsize x 2(Max_Program_Length) x 3(Max_Arguments + 1 for Function)]
                answer (tensor): Encoded Answers/Ground Truth Labels [Batchsize x 1]
        """
        self.im_height = height
        self.im_width = width
        self.im_dataroot = os.path.join(dataroot, 'Images')
        self.qs_dataroot = os.path.join(dataroot, 'Questions')
        self.args_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Arguments_Vocab_New.json'), 'r'))
        self.func_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Functions_Vocab_New.json'), 'r'))
        self.ans_vocab = json.load(open(os.path.join(dataroot, 'Vocabs', 'Answers_Vocab.json'), 'r'))
        self.partition = partition
        self.preprocess_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.Resize((self.im_height,self.im_width)),
                                                    transforms.ToTensor()])
        self.preprocess_val_test = transforms.Compose([transforms.Resize((self.im_height,self.im_width)),
                                                       transforms.ToTensor()])
        self.question_list = []
        self.question_type_list = []
        self.program_list = []
        self.image_list = []
        self.answer_list = []

        if partition == 'Train':
            question_file = open(os.path.join(self.qs_dataroot,'Train_Questions_Programs_New.json'))
            self.im_dataroot = os.path.join(self.im_dataroot, 'Train_Image')
        #elif partition == 'Val':
        #    question_file = open(os.path.join(self.qs_dataroot,'Valid_Question_Programs.json'))
        #    self.im_dataroot = os.path.join(self.im_dataroot, 'Valid_Image')
        elif partition == 'Test':
            question_file = open(os.path.join(self.qs_dataroot,'Test_Questions_Programs_New.json'))
            self.im_dataroot = os.path.join(self.im_dataroot, 'Train_Image')
        data = json.load(question_file)
        
        for key in list(data.keys()):
            q = data[key]['Question']
            q_type = data[key]['Question_Type']
            program = data[key]['Program']
            img_file = data[key]['Image_ID']
            ans = data[key]['Ground_Truth']
            self.question_list.append(q)
            self.question_type_list.append(q_type)
            self.program_list.append(program)
            self.image_list.append(img_file)
            self.answer_list.append(ans)

    def __len__(self):
        return len(self.image_list)

    def _prog_string_to_list(self, prog_string):
        """Returns formatted list version of Program String for Encoding
        """
        prog_string = prog_string.split(' & ')
        prog_list = []
        for i in prog_string:
            dict = {}
            prog = 'None'
            vals = 'None'
            if '(' in i:
                prog = i[:i.index('(')]
                vals = i[i.index('(')+1:i.index(')')].split(', ')
            else:
                prog = i
            dict['function'] = prog
            dict['argument'] = vals
            prog_list.append(dict)
        return prog_list

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Image
        image = Image.open(os.path.join(self.im_dataroot, self.image_list[index])).convert('RGB')
        if self.partition == 'train':
            image = self.preprocess_train(image).type(torch.FloatTensor)
        else:
            image = self.preprocess_val_test(image).type(torch.FloatTensor)
        
        # Question
        question = self.question_list[index]
        # Question Type
        qs_type = self.question_type_list[index]
        
        # Program
        program_string = self.program_list[index]
        program = self._prog_string_to_list(program_string)
        args = np.zeros((2, 3))
        num_progs = 0
        for p in program:
            func = p['function']
            arg_list = p['argument']
            if arg_list != 'None':
                for i in range(len(arg_list)):
                    arg_idx = self.args_vocab[arg_list[i]]
                    args[num_progs][i+1] = arg_idx
            func_idx = self.func_vocab[func]
            args[num_progs][0] = func_idx
            num_progs = num_progs + 1
        args = torch.as_tensor(args).long()
        # Answer
        answer = self.answer_list[index]
        answer = self.ans_vocab[str(answer)]
        return image, question, qs_type, program_string, args, answer