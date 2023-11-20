from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json


#model_name_or_path = "roberta-large"
model_name_or_path = "roberta-large"
dataset_name = "semeval"




tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
def split_label_words(tokenizer, label_list):
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
    t = json.load(file)
    label_list = list(t)

t = split_label_words(tokenizer, label_list)


model_name_or_path = "bart-large"
with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
    torch.save(t, file)


'''
每一行表示一个label对应的word id
tensor([[ 2362,  9355,     0,     0,     0,     0,     0,     0],
        [17247,  1938,   453,     0,     0,     0,     0,     0],
        [ 5970, 10384,     0,     0,     0,     0,     0,     0],
        [ 5970, 17117,     0,     0,     0,     0,     0,     0],
        [17247,  1938,   247,     9,  6084,     0,     0,     0],
        [ 5970,   247,     9,   744,     0,     0,     0,     0],
        [ 5970,  1041,     0,     0,     0,     0,     0,     0],
        [ 5970,   194,   368,  4892,  6320,  4643,     9,  5238],
        [17247,  1938,   299,   453,    73, 32198,  5421,     0],
        [17247,  1938, 25385,     0,     0,     0,     0,     0],
        [17247,  1938,   346,     9,  1321,    73, 23742,     0],
        [ 5970,   194,   368, 13138, 15062,     9,   744,     0],
        [ 5970,  9813,     0,     0,     0,     0,     0,     0],
        [ 5970,   408,     0,     0,     0,     0,     0,     0],
        [17247,  1938,   559,    73, 32300, 23114,     0,     0],
        [ 5970,   343,     9,  3113,     0,     0,     0,     0],
        [ 5970,  1270,     0,     0,     0,     0,     0,     0],
        [17247,  1938,  4071,     0,     0,     0,     0,     0],
        [ 5970,  3200,     9,     0,     0,     0,     0,     0],
        [17247,  1938,   919,     9,     0,     0,     0,     0],
        [17247,  1938,  4790,    30,     0,     0,     0,     0],
        [ 5970,   749,     9,  5238,     0,     0,     0,     0],
        [ 5970,    97,   284,     0,     0,     0,     0,     0],
        [ 5970,  6825,     0,     0,     0,     0,     0,     0],
        [ 5970,  3599,     0,     0,     0,     0,     0,     0],
        [ 5970,  1248,     9,  3113,     0,     0,     0,     0],
        [17247,  1938,   343,     9,  6084,     0,     0,     0],
        [17247,  1938, 14417,  2523,     0,     0,     0,     0],
        [17247,  1938,   998,     0,     0,     0,     0,     0],
        [ 5970,  1303,     9,   744,     0,     0,     0,     0],
        [17247,  1938,   194,   368, 13138, 15062,     9,  6084],
        [ 5970,  1304,  2922,     0,     0,     0,     0,     0],
        [ 5970,   247,     9,  3113,     0,     0,     0,     0],
        [ 5970,  1248,     9,   744,     0,     0,     0,     0],
        [ 5970,   343,     9,   744,     0,     0,     0,     0],
        [17247,  1938,  4790,     0,     0,     0,     0,     0],
        [ 5970,  1947,     9,  5238,     0,     0,     0,     0],
        [ 5970,  1046,     0,     0,     0,     0,     0,     0],
        [ 5970,  1103,     0,     0,     0,     0,     0,     0],
        [ 5970,   194,   368, 13138, 15062,     9,  3113,     0]])
'''