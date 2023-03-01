# TMN-FloodNet-VQA
Transformer Module Networks for Post-Flood Damage Assessment through Visual Question Answering

# Before Running TMN for FloodVQA
This codebase is built on top of Transformer Module Networks (Repository: https://gitlab.com/llml-mit/modular_transformer).
1) Clone this current Repository.
3) Download FloodNet Data - https://drive.google.com/drive/folders/1g1r419bWBe4GEF-7si5DqWCjxiC8ErnY?usp=sharing
4) Data Folder should have three sections
   - Images (Train_Image, Val_Image, Test_Image)
   - Questions (All Questions proecessed through Program Generator) - Can be downloaded from this repo
   - Vocabs (Function Vocabs, Argument Vocabs, Answer Vocabs) - Can be downloaded from this repo
5) Note that, when modularity of program changes then "args" matrix in DataLoader.py must change (num_prog_length, num_args)

# Execute Training TMN-FloodVQA
- For Linux based OS: \
`bash TMN_FloodNet_VF_Trainer.sh`
- For Windows based OS:\
`python './TMN_FloodNet_VF_Trainer.py' --learning_rate 1e-5 --save_name 'FloodNet_VF_Test' --im_height 64 --im_width 64 --num_epochs 20.0 --batch_size 32 --gas 1 --num_module_layers 1 --arch 's' --vf 'vt'`
- Note: Change any parameters as needed

# Execute Evaluation TMN-FloodVQA
- For Linux based OS: \
`bash TMN_FloodNet_VF_Evaluator.sh`
- For Windows based OS:\
`python './TMN_FloodNet_VF_Evaluator.py' --from_pretrained './TMN_FloodNet/Results/FloodNet_VT/Model/FloodNet_VF_Train_20/TMN_FloodNet_L1_Ep20.bin' --im_height 64 --im_width 64 --seed 1 --batch_size 32 --num_module_layers 1 --arch 's' --vf 'vt'`
- Note: Change any parameters as needed

# Note:
- This repository is still in Development Phase
