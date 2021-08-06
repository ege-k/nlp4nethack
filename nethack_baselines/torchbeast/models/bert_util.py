import torch
from NetHackCorpus.corpus_loader import load_corpus

from transformers import BertModel, BertTokenizer


def corpus_as_bert_embeddings(corpus_dir, device, output_device="cpu", cleaned=False):
    if cleaned:
        corpus = load_corpus(corpus_dir, cleaned=cleaned)
    else:
        manual, wiki, spoilers = load_corpus(corpus_dir, cleaned=cleaned)
        corpus = manual + wiki + spoilers

    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    filename_to_bert = dict()
    for doc in corpus:
        name = doc.split(" ")[0]

        input_ids = tokenizer.encode(doc)[:512]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        pooled_output = model(input_ids).pooler_output  # vec of size 1 x 768
        filename_to_bert[name] = pooled_output.to(output_device)

    return filename_to_bert


if __name__ == "__main__":
    from time import time
    start = time()
    ftb = corpus_as_bert_embeddings(corpus_dir="../NetHackCorpus/", device="cpu", output_device="cpu", cleaned=False)
    print("Loading the embeddings took:", time()-start)
    print(ftb.keys())
    print(list(ftb.keys())[0], ftb[list(ftb.keys())[0]])
