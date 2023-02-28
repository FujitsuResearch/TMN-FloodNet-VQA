import sys
sys.path.append(r"C:\Users\akradiptad\OneDrive - FUJITSU\Desktop\Work\BIG - FRJ and MIT\FloodNet\Code")

import os
import math
import random
import logging
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score

import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer)

from TMN_FloodNet.Config import PATH
from TMN_FloodNet.DataLoader import FloodNetVQA
from TMN.models.module_vf import TransformerModuleNetWithExtractor

logging.basicConfig(format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)s %(funcName)s %(message)s",
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Dataset Arguments
    parser.add_argument("--im_height", default=64, type=int, help="image height")
    parser.add_argument("--im_width", default=64, type=int, help="image width")
    # Evaluation Arguments
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int, help="training batch size")
    # Model Arguments
    parser.add_argument("--from_pretrained", default='', type=str, help="model path")
    parser.add_argument("--num_module_layers", type=int, default=1, help="Number of module layers.")
    parser.add_argument("--arch", type=str, default='s', help="Network architecture (stack, tree)", choices=['s', 't'])
    parser.add_argument("--vf", type=str, default='vt', help="use othre visual features", choices=['region', 'vt'])
    args = parser.parse_args()
    
    print('Arguments: ')
    print(args)
    print("Importing path Configs")

    path_cfgs = PATH()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    config = BertConfig.from_pretrained(path_cfgs.config_file)
    config.num_module_layer = args.num_module_layers
    config.arch = args.arch
    config.vf = args.vf
    config.use_location_embed = True
    
    print('Bert Model Configuration: ')
    print(config)
    print("Number of Module Layers: ", config.num_module_layer)
    print("Visual Features: ", 'Visual-Tokenizer' if args.vf == 'vt' else 'region')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    print('Seeds set for Reproducbility.')

    if args.arch == 's':
        from TMN.models.module_s import TransformerModuleNet
        print("Selected Architecture:  Stacked TMN")
    elif args.arch == 't':
        from TMN.models.module_t import TransformerModuleNet
        print("Selected Architecture:  Tree TMN")
        print('Tree TMN not Compatible with FloodNet')
        exit()
    else:
        print("Architecture should be [s, t], for FloodNet only stack")
        exit()

    # Set Up Model for Training
    extractor = None
    if not args.vf:
        print("use pre-trained object detector")
        exit()
    else:
        if args.vf == 'region':
            print("Selected Feature Extractor: Regional features without Visual Genome")
            print("Please select Visual Tokenizer")
            exit()
        elif args.vf == 'vt':
            from TMN.models.visual_tokenizer import VisualTokenizer
            extractor = VisualTokenizer(config)
            print("Selected Feature Extractor: Grid Features as Tokens with Visual Tokenizer")
            if args.arch == 's':
                config.num_region = 151
            elif args.arch == 't':
                config.num_region = 302
            config.use_location_embed = False    

    transformer = TransformerModuleNet(config, 
                                       num_modules = 6,  # Number of Module corresponding to each function (typically equals to num_progs) 
                                       max_prog_seq = 2, # Maximum Program Sequence
                                       num_progs = 6,    # Number of different programs
                                       num_args = 6,     # Number of different arguments
                                       num_labels = 41)  # Number of Possible Answers in Entire Dataset
    model = TransformerModuleNetWithExtractor(config, transformer, extractor)

    if args.from_pretrained:
        model.load_state_dict(torch.load(args.from_pretrained))
        print(f'Loaded Pretrained Model From {args.from_pretrained}')

    model.to(device)
    print('Model Initialized to: ',device)
    
    dataset = FloodNetVQA(dataroot = path_cfgs.dataset_path,
                          partition = 'Train',
                          height = args.im_height,
                          width = args.im_width)
    #print('Training Dataset Loaded')

    TRAIN_DATA_LEN = 3000
    VAL_DATA_LEN = 500
    TEST_DATA_LEN = len(dataset) - (TRAIN_DATA_LEN + VAL_DATA_LEN)

    _, _, test_dataset = random_split(dataset, [TRAIN_DATA_LEN, VAL_DATA_LEN, TEST_DATA_LEN])
    print("Test Data Examples:",len(test_dataset)) 

    test_data_loader = DataLoader(test_dataset, 
                                  batch_size = args.batch_size, 
                                  num_workers = 4)
                    
    logger.info('Model Testing Started')
    torch.set_grad_enabled(False)

    eval_total_matches = 0
    eval_total_loss = 0

    total_simple = 0
    total_complex = 0
    total_yesno = 0
    total_condition = 0
    eval_simple_matches = 0
    eval_complex_matches = 0
    eval_yesno_matches = 0
    eval_condition_matches = 0
    y_true_total = []
    y_pred_total = []
    y_true_simple = []
    y_pred_simple = []
    y_true_complex = []
    y_pred_complex = []
    y_true_yesno = []
    y_pred_yesno = []
    y_true_condition = []
    y_pred_condition = []

    for step, batch in enumerate(test_data_loader):
        if device != torch.device("cpu"):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        img, qs, qs_type, pg_str, arguments, answer_id = (batch)
        regions, img_info, spatials, image_mask  = None, None, None, None  # Visual Tokenizer does not use these arguments
        #spatials = np.zeros((self.region_len, 5))
        
        img = img.type(torch.FloatTensor).to(device)
        arguments = arguments.type(torch.LongTensor).to(device)
        answer_id = answer_id.type(torch.LongTensor).to(device)
        _, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, answer_id)

        if n_gpu > 1:
            loss = loss.mean() 

        logits = torch.max(pred, 1)[1].data  # argmax

        # Total Accuracy Calculation
        count_matches = ((logits - answer_id) == 0).sum().float()
        eval_total_matches += count_matches.item()   
        eval_total_loss += loss.item()

        # Question Typewise Score Calculation
        for id in range(len(qs_type)):
            y_true_total.append(answer_id[id])
            y_pred_total.append(logits[id])
            if qs_type[id] == 'Simple_Counting':
                total_simple += 1
                if logits[id] == answer_id[id]:
                    eval_simple_matches += 1
                y_true_simple.append(answer_id[id])
                y_pred_simple.append(logits[id])
            if qs_type[id] == 'Complex_Counting':
                total_complex += 1
                if logits[id] == answer_id[id]:
                    eval_complex_matches += 1
                y_true_complex.append(answer_id[id])
                y_pred_complex.append(logits[id])
            if qs_type[id] == 'Yes_No':
                total_yesno += 1
                if logits[id] == answer_id[id]:
                    eval_yesno_matches += 1
                y_true_yesno.append(answer_id[id])
                y_pred_yesno.append(logits[id])
            if qs_type[id] == 'Condition_Recognition':
                total_condition += 1
                if logits[id] == answer_id[id]:
                    eval_condition_matches += 1
                y_true_condition.append(answer_id[id])
                y_pred_condition.append(logits[id])

    eval_acc = eval_total_matches / float(TEST_DATA_LEN)
    eval_f1 = f1_score(y_true_total, y_pred_total, average='weighted')
    eval_prec = precision_score(y_true_total, y_pred_total, average='weighted')
    eval_simple_acc = eval_simple_matches / float(total_simple)
    eval_simple_f1 = f1_score(y_true_simple, y_pred_simple, average='weighted')
    eval_simple_prec = precision_score(y_true_simple, y_pred_simple, average='weighted')
    eval_complex_acc = eval_complex_matches / float(total_complex)
    eval_complex_f1 = f1_score(y_true_complex, y_pred_complex, average='weighted')
    eval_complex_prec = precision_score(y_true_complex, y_pred_complex, average='weighted')
    eval_yesno_acc = eval_yesno_matches / float(total_yesno)
    eval_yesno_f1 = f1_score(y_true_yesno, y_pred_yesno, average='weighted')
    eval_yesno_prec = precision_score(y_true_yesno, y_pred_yesno, average='weighted')
    eval_condition_acc = eval_condition_matches / float(total_condition)
    eval_condition_f1 = f1_score(y_true_condition, y_pred_condition, average='weighted')
    eval_condition_prec = precision_score(y_true_condition, y_pred_condition, average='weighted')
    eval_loss = eval_total_loss / float(TEST_DATA_LEN)

    print('Test Results:') 
    print(f'Total::                         Accuracy: {"{:.4f}".format(eval_acc)}       F1 Score: {"{:.4f}".format(eval_f1)}        Precision: {"{:.4f}".format(eval_prec)}', flush=True)
    print(f'Simple Counting::               Accuracy: {"{:.4f}".format(eval_simple_acc)}       F1 Score: {"{:.4f}".format(eval_simple_f1)}        Precision: {"{:.4f}".format(eval_simple_prec)}', flush=True)
    print(f'Complex Counting::              Accuracy: {"{:.4f}".format(eval_complex_acc)}       F1 Score: {"{:.4f}".format(eval_complex_f1)}        Precision: {"{:.4f}".format(eval_complex_prec)}', flush=True)
    print(f'Yes-No Question                 Accuracy: {"{:.4f}".format(eval_yesno_acc)}       F1 Score: {"{:.4f}".format(eval_yesno_f1)}        Precision: {"{:.4f}".format(eval_yesno_prec)}', flush=True)
    print(f'Condition-Recongnition Question Accuracy: {"{:.4f}".format(eval_condition_acc)}       F1 Score: {"{:.4f}".format(eval_condition_f1)}        Precision: {"{:.4f}".format(eval_condition_prec)}', flush=True)
    print(f'Total Cross-Entropy::               Loss: {"{:.4f}".format(eval_loss)}', flush=True)
    logger.info('Model Testing Finished')
    print('Experiment Finished')

if __name__ == "__main__":
    main()