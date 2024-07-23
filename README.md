# GAP: A novel Generative context-Aware Prompt-tuning method for relation extraction

**The code of this repository is constantly being updated...**

**Our code is based on the [KnowPrompt](https://github.com/zjunlp/KnowPrompt) .**

Our code consists of three crucial modules: 
  1. A pretrained prompt generator module that extracts or generates the relation triggers from the context and embeds them into the prompt tokens;
  2. An in-domain adaptive pretraining module that further trains the Pretrained Language Models (PLMs) to promote the adaptability of the model;
  3. A joint contrastive loss that prevents PLMs from generating unrelated content and optimizes our model more effectively.



Please look forward to it!

## Introduction

This repository is used in our paper:

《GAP: A novel Generative context-Aware Prompt-tuning method for Relation Extraction》

Zhenbin Chen, Zhixin Li, Ying Huang, Zhenjun Tang. 


Please cite our paper and kindly give a star for this repository if you use this code.


## Usage

### Semeval
```
sh ./scripts/semeval.sh
```

### TACRED
```
sh ./scripts/tacred.sh
```

### TACREV
```
sh ./scripts/tacrev.sh
```

### Re-TACRED
```
sh ./scripts/retacred.sh
```

## Citation
```
@article{chen2024gap,
  title={GAP: A novel Generative context-Aware Prompt-tuning method for relation extraction},
  author={Chen, Zhenbin and Li, Zhixin and Zeng, Yufei and Zhang, Canlong and Ma, Huifang},
  journal={Expert Systems with Applications},
  volume={248},
  pages={123478},
  year={2024},
  publisher={Elsevier}
}
```


