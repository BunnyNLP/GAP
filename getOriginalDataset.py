from fileinput import filename
import json,os
from pathlib import Path

dataset_name = "tacrev"

with open(f"./dataset/{dataset_name}/k-shot/8-1/train.txt",'r',encoding='utf-8') as fr:
    data = fr.readlines()

print(type(data))
data = [" ".join(eval(x)['token'])+"\n" for x in data]
print(len(data))

# filename = Path(f"/home/jxlab/workspace/czb/MyPrompt/dataset/k-shot/8-1/{dataset_name}.txt")
# filename.touch(exist_ok=True)  # will create file, if it exists will do nothing
with open(f"./dataset/{dataset_name}/k-shot/8-1/{dataset_name}.txt",'w',encoding='utf-8') as fw:
    fw.writelines(data)
