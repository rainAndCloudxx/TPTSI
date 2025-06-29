# TPTSI

## Dataset

### Detailed statistics of benchmark dataset

| Dataset         | #Tar    | %M   | #Sent  | #Len    |
|-----------------|---------|------|--------|---------|
| MOH(train)      | 1,489   |22.49 | 1,482  | 7.31    |
| MOH(test)       | 150     |50.00 | 150    | 7.26    |
| MOH(val)        | 210     |50.00 | 70     | 7.17    |
| TroFi(train)    | 1,772   |52.65 | 1,697  | 28.06   |
| TroFi(test)     | 1,965   |61.67 | 1,925  | 29.05   |
| TroFi(val)      | 650     |60.92 | 647    | 28.88   |
| VUA(train)      | 160,154 |11.97 | 10,909 | 27.59   |
| VUA(test)       | 22,196  |17.94 | 3,601  | 28.1    |
| VUA(val)        | 6,658   |17.99 | 1,996  | 28.39   |

<b>MOH-X<b> is sourced from WordNet and focuses on annotating metaphorical and literal verb usages. Each verb has multiple usages, including at least one metaphorical one. 
<b>TroFi<b> is from the Wall Street Journal corpus (1987–1989) and also focuses on verb usages. Compared with MOH-X, TroFi’s data source is closer to real scenarios, mostly from formal news texts. 
<b>VUA<b> covers various text types and annotates metaphors for different parts of speech. <b>VUA-20<b>, an extended version by Leong et al. in 2020, continues the VUA annotation style and expands the data scale and coverage.

<br><br>
<!-- Preprocessed Datasets -->

You can get datasets from the following [link](https://drive.google.com/file/d/18hemekvUuOw-qkQhWv4F6qFnis6lkaog/view?usp=sharing).

The datasets are tsv formatted files and the format is as follows.
```
index	label	sentence	POS	w_index
a3m-fragment02 45	0	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	NOUN	0
a3m-fragment02 45	1	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	ADJ	1
a3m-fragment02 45	1	Design: Crossed lines over the toytown tram: City transport could soon be back on the right track, says Jonathan Glancey	NOUN	2
```

You can also get the original datasets from the following links:

<!-- MOH-X -->
- MOH-X: [https://github.com/RuiMao1988/Sequential-Metaphor-Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification)

<!-- TroFi -->
- TroFi: [https://github.com/RuiMao1988/Sequential-Metaphor-Identification](https://github.com/RuiMao1988/Sequential-Metaphor-Identification)
  
<!-- VUA-18 and VUA-20 -->

- VUA-20: [https://github.com/YU-NLPLab/DeepMet](https://github.com/YU-NLPLab/DeepMet)

<br>

## Basic Usage
- Change the experimental settings in `config/config.yaml`. <br>
- Run `main.py` to train and test models. <br>
- Command line arguments are also acceptable with the same naming in configuration files.
- Download Pytorch RoBERTa model from Huggingface https://huggingface.co/roberta-base and put in the folder `roberta-base`.
  
## Running MelBERT

Run the following command for training:<br>
`python main.py`

- Using RoBERTa, MelBERT gets about 78.5 and 75.7 F1 scores on the VUA-18 and the VUA-verb set, respectively. Using model bagging techniques, we get about 79.8 and 77.1 F1 scorea on the VUA-18 and VUA-verb set, respectively.

