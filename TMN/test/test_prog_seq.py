import json
import torch
import numpy as np

from cfgs.path_cfgs import PATH


def main():
    print("import path cfgs")
    path_cfgs = PATH()
    feature_path = path_cfgs.corpus_path_val
    # question_path = path_cfgs.caption_path_val
    answer_vocab_path = path_cfgs.vocab_path_vilbert_clevr_elemental
    args_vocab_path = path_cfgs.args_vocab_path_vilbert_clevr_elemental
    func_vocab_path = path_cfgs.func_vocab_path_vilbert_clevr_elemental

    # annos = json.load(open(question_path, 'r'))
    # ques = annos["questions"]

    # print(len(ques))
    # q = ques[0]

    sample_qustion_path = path_cfgs.sample_qustion_path

    q = json.load(open(sample_qustion_path, 'r'))

    func_vocab = json.load(open(func_vocab_path, 'r'))
    args_vocab = json.load(open(args_vocab_path, 'r'))

    progs = q["program"]

    print(q)
    print(progs)

    args = np.full((25, 2), 20)
    num_progs = 0
    for p in progs:
        func = p['function']
        arg = p['value_inputs']
        arg_idx = 19
        if arg:
            arg_idx = args_vocab[arg[0]]
        else:
            arg = 'none'

        func_idx = func_vocab[func]
        # print(func, func_idx, "|", arg, arg_idx)
        if func_idx > 0:
        #     args.append([func_idx, arg_idx])
            args[num_progs] = [func_idx, arg_idx]
            num_progs = num_progs + 1

    # print(args)
    print(num_progs)

    args_tensor = torch.tensor(args)

    print(args_tensor)
    print(args_tensor.shape)


if __name__ == "__main__":
    main()