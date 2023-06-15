# Fair and private language models

This repository contains the code for our article accepted for publication at the Findings of the 61th
Annual Meeting of the Association for Computational Linguistics (Findings of ACL 2023): [Trade-Offs Between Privacy and Fairness in Language Models](https://arxiv.org/pdf/2305.14936.pdf).

> **Abstract**: 
> Protecting privacy in contemporary NLP models is gaining in importance. So does the need to mitigate social biases of such models. But can we have both at the same time? Existing research suggests that privacy preservation comes at the price of worsening biases in classification tasks. In this paper, we explore the extent to which this tradeoff really holds when we incorporate both privacy preservation and debiasing techniques into training text generation models. How does improving the model along one dimension affect the other dimension as well as the utility of the model? We conduct an extensive set of experiments that include bias detection, privacy attacks, language modeling, and performance on downstream tasks.

## File Structure and code usage
The focus of this work was to train different models with either debiasing, privacy, or both objectives and evaluate the privacy and/or bias in the resulting models. We further evaluated the language modeling ability and performance on downstream tasks.

The order of execution of our code is therefore 
1. Create the dataset in [data_prep](https://github.com/cleolotta/fair-and-private-lm/tree/main/data_prep)
2. Train the models and perform membership inference attacks in [code](https://github.com/cleolotta/fair-and-private-lm/tree/main/code)
3. Evaluate the models for bias and language modeling ability in [bias-evaluation](https://github.com/cleolotta/fair-and-private-lm/tree/main/bias-evaluation)
4. Evaluate the downstream performance in [glue](https://github.com/cleolotta/fair-and-private-lm/tree/main/glue)

Each folder contains a description and instructions on how to use the included codes.

```bigquery
@article{matzken2023trade,
  title="Trade-Offs Between Fairness and Privacy in Language Modeling",
  author="Matzken, Cleo and Eger, Steffen and Habernal, Ivan",
  journal="Findings of the Association for Computational Linguistics: ACL 2023",
  year = "2023",
  url = "https://arxiv.org/abs/2305.14936",
  note = "accepted"
}
```

