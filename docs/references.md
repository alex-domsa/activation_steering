# References

Papers this project builds on. PDFs are kept locally in `docs/papers/` (not committed).

## Theory of analogy
- **Gentner, D. (1983).** *Structure-mapping: A theoretical framework for analogy.* Cognitive Science, 7(2), 155-170.
  https://doi.org/10.1207/s15516709cog0702_3
- **Gentner, D. & Markman, A. B. (1997).** *Structure mapping in analogy and similarity.* American Psychologist, 52(1), 45-56.
  https://doi.org/10.1037/0003-066X.52.1.45

  Structure-mapping theory: an analogy is a match in relational structure (how things relate), not in object attributes
  (what the things are). Literal similarity shares both; analogy shares relations while the objects differ. This is why the
  project contrasts *analogy* (high relation, low entity similarity) with *unrelated* pairs (low on both), and why
  *similar* pairs (high on both) are not used as the contrast: they differ from analogies only in entity similarity.

## Method
- **Rimsky et al. (2024).** *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. https://arxiv.org/abs/2312.06681
  The CAA method: a steering vector is the mean activation difference between contrasting examples, added at one layer.
- **Todd et al. (2024).** *Function Vectors in Large Language Models.* ICLR 2024. https://arxiv.org/abs/2310.15213
  A vector for a task (e.g. country -> capital), extracted from in-context examples, makes the model do the task zero-shot.
- **Hendel, Geva & Globerson (2023).** *In-Context Learning Creates Task Vectors.* Findings of EMNLP 2023. https://arxiv.org/abs/2310.15916
  Patching the hidden state of the "->" token transfers the task; the basis of experiment 09.
- **Marks & Tegmark (2023).** *The Geometry of Truth.* https://arxiv.org/abs/2310.06824
  Mean-difference directions, generalisation checks and causal interventions for a "truth" direction.

## Reliability of steering vectors
- **Tan et al. (2024).** *Analysing the Generalisation and Reliability of Steering Vectors.* https://arxiv.org/abs/2407.12404
  Steering vectors often carry spurious biases (e.g. towards "Yes" or "A") and vary across inputs.
- **Braun et al. (2025).** *Understanding (Un)Reliability of Steering Vectors in Language Models.* https://arxiv.org/abs/2505.22637
  Steering works when per-example activation differences are coherent and separable; the go/no-go check in experiment 08.

## Analogy in language models
- **Jiayang et al. (2023).** *StoryAnalogy.* EMNLP 2023. https://arxiv.org/abs/2310.12874
- ***Analogical Reasoning Inside Large Language Models: Concept Vectors and the Limits of Abstraction* (2025).** https://arxiv.org/abs/2503.03666
  Concept vectors from attention heads are invariant for verbal concepts but not for abstract ones such as "previous"/"next".
- ***The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models* (2025).** https://arxiv.org/abs/2511.20344
  Uses StoryAnalogy with Qwen2.5-14B; analogy is linearly decodable in middle layers; no steerable analogy direction is claimed.

## Tools
- **Fiotto-Kaufman et al. (2025).** *NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals.* ICLR 2025.
  https://arxiv.org/abs/2407.14561
