# Data

The datasets are not stored in this repository. Download them into this `data/` folder.

## StoryAnalogy (main dataset)
Jiayang et al., *StoryAnalogy: Deriving Story-level Analogies from Large Language Models to Unlock
Analogical Understanding*, EMNLP 2023. Code and data: https://github.com/loginaway/StoryAnalogy (MIT licence).

Expected file: `data/StoryAnalogy.csv` with columns `s1, s2, entity, relation, domain`
(24,388 story pairs; `entity` and `relation` are mean human ratings on a 0-3 scale).

## sentence_analogy (A : B :: C : D analogies, used in experiment 09)
`liyannn/sentence_analogy` on Hugging Face. It is **gated**: accept the conditions on the dataset page and log in with
`huggingface-cli login`. Experiment 09 downloads `mixgoogle-annotated.tsv` automatically
(Google analogy word pairs placed in sentence templates). Do not commit the file.
