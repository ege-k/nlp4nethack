import pickle

import torch
from NetHackCorpus.corpus_loader import load_corpus
from tqdm import trange
import psutil

from transformers import BertModel, BertTokenizer, logging

def load_ftb_from_file(file="ftb.pkl"):
    with open(file, "rb") as f:
        ftb = pickle.load(f)
    return ftb

def save_ftb_in_file(ftb, file="ftb.pkl"):
    with open("ftb.pkl", "wb") as f:
        pickle.dump(ftb, f)

def corpus_as_bert_embeddings(corpus_dir, device, output_device="cpu", cleaned=False):

    print("Available space:", psutil.virtual_memory().total)

    free = psutil.virtual_memory().available
    if cleaned:
        corpus = load_corpus(corpus_dir, cleaned=cleaned)
    else:
        manual, wiki, spoilers = load_corpus(corpus_dir, cleaned=cleaned)
        corpus = manual + wiki + spoilers
        del manual
        del wiki
        del spoilers
    corpus_size = free - psutil.virtual_memory().available

    logging.set_verbosity_error()

    free = psutil.virtual_memory().available
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_size = free - psutil.virtual_memory().available

    print("Corpus takes:", corpus_size, "Bert takes", bert_size)

    filename_to_bert = dict()
    t = trange(len(corpus), desc='Bar desc', leave=True)
    free = psutil.virtual_memory().available

    for idx in t:
        doc = corpus[idx]
        name = doc.split(" ")[0]

        input_ids = tokenizer.encode("[CLS] " + doc)[:512]
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            pooled_output = model(input_ids).pooler_output  # vec of size 1 x 768
        filename_to_bert[name] = pooled_output.to(output_device)

        free_2 = psutil.virtual_memory().available
        t.set_description("Bar desc (file %i), average size: %i" % (idx, (free-free_2)/(idx+1)))
        t.refresh()  # to show immediately the update

    return filename_to_bert


if __name__ == "__main__":
    from time import time
    start = time()
    ftb = corpus_as_bert_embeddings(corpus_dir="../NetHackCorpus/", device="cpu", output_device="cpu", cleaned=False)
    print("Loading the embeddings took:", time()-start)
    print(ftb.keys())
    print(list(ftb.keys())[0], ftb[list(ftb.keys())[0]])
