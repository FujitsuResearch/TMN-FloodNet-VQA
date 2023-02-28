from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from vilbert.datasets.clevr_dataset import CLEVRDataset

from cfgs.path_cfgs import PATH


def main():
    print("import path cfgs")
    path_cfgs = PATH()
    corpus_path = path_cfgs.corpus_path_val
    caption_path = path_cfgs.caption_path_val
    vocab_path = path_cfgs.vocab_path_vilbert_clevr

    bert_model = path_cfgs.bert_model
    tokenizer = BertTokenizer.from_pretrained(
        bert_model, do_lower_case=True
    )

    dataset = CLEVRDataset(
        corpus_path,
        caption_path,
        vocab_path,
        tokenizer,
        seq_len=36,
    )

    data_loader = DataLoader(dataset, batch_size=16, num_workers=0)

    for step, batch in enumerate(data_loader):
        print(step)
        exit()


if __name__ == "__main__":
    main()