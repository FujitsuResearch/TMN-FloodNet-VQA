import sys
sys.path.append(r"C:\Users\akradiptad\OneDrive - FUJITSU\Desktop\Work\BIG - FRJ and MIT\FloodNet\Code")

import os
import math
import random
import logging
import argparse
import numpy as np

import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer)

from TMN_FloodNet.Config import PATH
from TMN_FloodNet.DataLoader import FloodNetVQA, Synthetic_Dataset
from TMN.models.module_vf import TransformerModuleNetWithExtractor

logging.basicConfig(format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)s %(funcName)s %(message)s",
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Dataset Arguments
    parser.add_argument("--save_name", default='FloodNet_Syn_Real', type=str, help="save name for training.")
    parser.add_argument("--start_epoch", default=0, type=float, help="start epoch.")
    parser.add_argument("--num_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="training batch size")
    parser.add_argument("--im_height", default=128, type=int, help="image height")
    parser.add_argument("--im_width", default=128, type=int, help="image width")
    # Training Arguments
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gas", default=1, type=float, help="gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # Model Arguments
    parser.add_argument("--num_module_layers", type=int, default=1, help="Number of module layers.")
    parser.add_argument("--from_pretrained", default='', type=str, help="model path")
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

    savePath = os.path.join(path_cfgs.save_path, args.save_name)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

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
                                       num_labels = 42)  # Number of Possible Answers in Entire Dataset
    model = TransformerModuleNetWithExtractor(config, transformer, extractor)

    if args.from_pretrained:
        model.load_state_dict(torch.load(args.from_pretrained))
        print(f'Loaded Transformer From {args.from_pretrained}')

    model.to(device)
    print('Model Initialized to: ',device)

    train_dataset = Synthetic_Dataset(dataroot = path_cfgs.dataset_path,
                                      height = args.im_height,
                                      width = args.im_width)
    
    test_dataset = FloodNetVQA(dataroot = path_cfgs.dataset_path,
                               partition = 'Train',
                               height = args.im_height,
                               width = args.im_width)
    
    TRAIN_DATA_LEN = 3000
    VAL_DATA_LEN = 500
    TEST_DATA_LEN = len(test_dataset) - (TRAIN_DATA_LEN + VAL_DATA_LEN)

    _, validation_dataset, test_dataset = random_split(test_dataset, [TRAIN_DATA_LEN, VAL_DATA_LEN, TEST_DATA_LEN])
    
    print("Synthetic Training Data Examples: ",len(train_dataset))
    print("Real Validation Data Examples   :",len(validation_dataset))
    print("Real Test Data Examples         :",len(test_dataset)) 
    print('Datasets Loaded')

    train_batch_size = args.batch_size
    val_batch_size = args.batch_size 
    num_train_epochs = args.num_epochs
    start_epoch = args.start_epoch
    gradient_accumulation_steps = args.gas
    
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size = train_batch_size, 
                                   num_workers = 16, 
                                   shuffle = True)
    validation_data_loader = DataLoader(validation_dataset, 
                                        batch_size = val_batch_size, 
                                        num_workers = 4)
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size = val_batch_size, 
                                  num_workers = 4)

    num_train_optimization_steps = (math.ceil(TRAIN_DATA_LEN / train_batch_size / gradient_accumulation_steps)* (num_train_epochs - start_epoch))

    # warmup_steps = num_train_optimization_steps * 0.1
    warmup_steps = 0
    learning_rate = args.learning_rate
    adam_epsilon = 1e-8

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], 
                                                  "lr": learning_rate, 
                                                  "weight_decay": 0.01}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], 
                                                  "lr": learning_rate, 
                                                  "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'Model Converted to MultiGPU Model. Using: {n_gpu} GPUs')

    # Train
    print('Training Information:')
    print(f'Num examples = {len(train_dataset)}')
    print(f'Num Epochs = {num_train_epochs}')
    print(f'Learning rate = {learning_rate}')
    print(f'Total train batch size (w. parallel, distributed & accumulation) = {train_batch_size * gradient_accumulation_steps}')
    print(f'Gradient Accumulation steps = {gradient_accumulation_steps}')
    print(f'Total optimization steps = {num_train_optimization_steps}')

    num_steps = int(TRAIN_DATA_LEN / train_batch_size / gradient_accumulation_steps)
    model.zero_grad()
    
    global_step = 0
    step_tmp = 0
    startIterID = 0
    matches_tmp = 0
    loss_tmp = 0
    global_loss_tmp = 0
    global_matches_tmp = 0
    best_eval_score = 0
    best_model = None

    logger.info('Model Training Started')
    torch.autograd.set_detect_anomaly(True)
    
    # Model Training
    for epochId in range(int(start_epoch), int(num_train_epochs)):
        model.train()
        matches_tmp = 0
        loss_tmp = 0
        step_tmp = 0
        global_loss_tmp = 0
        global_matches_tmp = 0

        for step, batch in enumerate(train_data_loader):
            iterId = startIterID + step + (epochId * num_steps)
            if device != torch.device("cpu"):
                batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            img, qs, qs_type, pg_str, arguments, answer_id = (batch)
            # Visual Tokenizer does not use these arguments
            regions, img_info, spatials, image_mask  = None, None, None, None  

            img = img.type(torch.FloatTensor).to(device)
            arguments = arguments.type(torch.LongTensor).to(device)
            answer_id = answer_id.type(torch.LongTensor).to(device)
            outputs, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            matches_tmp += count_matches.item()
            loss_tmp += loss.item()
            step_tmp += 1

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == int(train_dataset.num_dataset / train_batch_size):
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # optimizer.zero_grad()
                scheduler.step()        # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                global_loss_tmp += loss_tmp
                global_matches_tmp += matches_tmp

                # print(epochId, step, global_step, matches_tmp / (step_tmp * train_batch_size), loss_tmp, " | ", scheduler.get_last_lr()[0], flush=True)
                # print(f'Epoch:{epochId}, Step:{step}, g:{global_step}, r:{matches_tmp / (step_tmp * train_batch_size)}, loss:{loss_tmp} | lr:{scheduler.get_last_lr()[0]}', flush=True)

                matches_tmp = 0
                loss_tmp = 0
                step_tmp = 0

                if global_step % 20 == 0 and global_step != 0:
                    global_loss_tmp = global_loss_tmp / 20.0
                    global_matches_tmp = global_matches_tmp / (gradient_accumulation_steps * train_batch_size * 20.0)

                    print(f'Training Epoch:{epochId}, Step:{step}, Global Step:{global_step}, Accuracy:{"{:.4f}".format(global_matches_tmp)}, Loss:{"{:.4f}".format(global_loss_tmp)} | lr:{scheduler.get_last_lr()[0]}', flush=True)

                    global_loss_tmp = 0
                    global_matches_tmp = 0

        # Validation after Each Epoch 
        torch.set_grad_enabled(False)
        model.eval()

        eval_total_matches = 0
        eval_total_loss = 0
        step_tmp_val = 0

        for step, batch in enumerate(validation_data_loader):
            iterId = startIterID + step + (epochId * num_steps)
            if device != torch.device("cpu"):
                batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            img, qs, qs_type, pg_str, arguments, answer_id = (batch)
            regions, img_info, spatials, image_mask  = None, None, None, None  # Visual Tokenizer does not use these arguments
            #spatials = np.zeros((self.region_len, 5))
            
            img = img.type(torch.FloatTensor).to(device)
            arguments = arguments.type(torch.LongTensor).to(device)
            answer_id = answer_id.type(torch.LongTensor).to(device)
            outputs, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

            

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            eval_total_matches += count_matches.item()   
            eval_total_loss += loss.item()
            step_tmp_val += img.size(0)

        eval_score = eval_total_matches / float(VAL_DATA_LEN)
        eval_loss = eval_total_loss / float(VAL_DATA_LEN)

        print(f'Validation Epoch:{epochId}, Accuracy:{eval_score}, Loss:{"{:.4f}".format(eval_loss)}', flush=True)
        torch.set_grad_enabled(True)

        # Save a trained model
        if eval_score >= best_eval_score:
            best_eval_score = eval_score
            best_model = model
            print("Model Save Policy: Best Validation Accuracy. Saving finetuned model at Epoch: ", epochId)
            model_to_save = (model.module if hasattr(model, "module") else model)  # Only save the model it-self
            output_model_file = os.path.join(savePath, "TMN_FloodNet_Syn_Real_L" + str(args.num_module_layers) + '_Ep' + str(int(args.num_epochs)) + ".bin")

        torch.save(model_to_save.state_dict(), output_model_file)

    logger.info('Finished Training and Validation')
    print('Best Validation Accuracy: ' + "{:.4f}".format(best_eval_score))
    
    logger.info('Model Testing Started')
    torch.set_grad_enabled(False)
    Inference_Model = best_model
    Inference_Model.to(device)
    Inference_Model.eval()
    print('Best Validation Model Loaded for Testing')

    eval_total_matches = 0
    eval_total_loss = 0
    step_tmp_val = 0

    for step, batch in enumerate(test_data_loader):
        iterId = startIterID + step + (epochId * num_steps)
        if device != torch.device("cpu"):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        img, qs, qs_type, pg_str, arguments, answer_id = (batch)
        regions, img_info, spatials, image_mask  = None, None, None, None  # Visual Tokenizer does not use these arguments
        #spatials = np.zeros((self.region_len, 5))
        
        img = img.type(torch.FloatTensor).to(device)
        arguments = arguments.type(torch.LongTensor).to(device)
        answer_id = answer_id.type(torch.LongTensor).to(device)
        outputs, pred =  Inference_Model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, answer_id)

        if n_gpu > 1:
            loss = loss.mean() 
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        logits = torch.max(pred, 1)[1].data  # argmax
        count_matches = ((logits - answer_id) == 0).sum().float()

        eval_total_matches += count_matches.item()   
        eval_total_loss += loss.item()
        step_tmp_val += img.size(0)

    eval_score = eval_total_matches / float(TEST_DATA_LEN)
    eval_loss = eval_total_loss / float(TEST_DATA_LEN)

    print(f'Testing Accuracy:{eval_score}, Loss:{"{:.4f}".format(eval_loss)}', flush=True)
    logger.info('Model Testing Finished')
    print('Experiment Finished')

if __name__ == "__main__":
    main()