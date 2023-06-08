# Fair and private language models

This repository contains the code for our article accepted for publication at the Findings of the 61th
Annual Meeting of the Association for Computational Linguistics (Findings of ACL 2023): [Trade-Offs Between Privacy and Fairness in Language Models](https://arxiv.org/pdf/2305.14936.pdf).

> **Abstract**: 
>Protecting privacy in contemporary NLP models is gaining in importance. So does the need to mitigate social biases of such models. But can we have both at the same time? Existing research suggests that privacy preservation comes at the price of worsening biases in classification tasks. In this paper, we explore the extent to which this tradeoff really holds when we incorporate both privacy preservation and debiasing techniques into training text generation models. How does improving the model along one dimension affect the other dimension as well as the utility of the model? We conduct an extensive set of experiments that include bias detection, privacy attacks, language modeling, and performance on downstream tasks.



### bias-evaluation
This folder contains the code to evaluate the models for a gender bias. The frameworks are SEAT (May et al., 2019), StereoSet (Nadeem et al.,) and BEC-Pro (Bartl et al., 2020).
##### Sources

SEAT:

@article{may2019measuring,
  title={On measuring social biases in sentence encoders},
  author={May, Chandler and Wang, Alex and Bordia, Shikha and Bowman, Samuel R and Rudinger, Rachel},
  journal={arXiv preprint arXiv:1903.10561},
  year={2019}
}

StereoSet:

@article{nadeem2020stereoset,
  title={Stereoset: Measuring stereotypical bias in pretrained language models},
  author={Nadeem, Moin and Bethke, Anna and Reddy, Siva},
  journal={arXiv preprint arXiv:2004.09456},
  year={2020}
}

BEC-Pro:

@article{bartl2020unmasking,
  title={Unmasking contextual stereotypes: Measuring and mitigating BERT's gender bias},
  author={Bartl, Marion and Nissim, Malvina and Gatt, Albert},
  journal={arXiv preprint arXiv:2010.14534},
  year={2020}
}

SEAT and StereoSet code:

@article{meade2021empirical,
  title={An empirical survey of the effectiveness of debiasing techniques for pre-trained language models},
  author={Meade, Nicholas and Poole-Dayan, Elinor and Reddy, Siva},
  journal={arXiv preprint arXiv:2110.08527},
  year={2021}
}

### code 
This folder contains the code to run the causal language modeling and to perform membership inference attack
##### Sources
source code with with own adjustments:

@article{mireshghallah2022memorization,
  title={Memorization in NLP Fine-tuning Methods},
  author={Mireshghallah, Fatemehsadat and Uniyal, Archit and Wang, Tianhao and Evans, David and Berg-Kirkpatrick, Taylor},
  journal={arXiv preprint arXiv:2205.12506},
  year={2022}
}

### data_prep
This folder contains the code used to prepare the dataset and the data to counterfactually augment the data in the course of causal language modeling.

##### Sources 
code with with own adjustments:

@article{lauscher2021sustainable,
  title={Sustainable modular debiasing of language models},
  author={Lauscher, Anne and L{\"u}ken, Tobias and Glava{\v{s}}, Goran},
  journal={arXiv preprint arXiv:2109.03646},
  year={2021}
}

Wordpairs:

@article{zhao2018gender,
  title={Gender bias in coreference resolution: Evaluation and debiasing methods},
  author={Zhao, Jieyu and Wang, Tianlu and Yatskar, Mark and Ordonez, Vicente and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:1804.06876},
  year={2018}
}


### glue
This folder contains the code used to evaluate the resulting models on the glue benchmark.

##### Sources
SEAT and StereoSet code:

@article{meade2021empirical,
  title={An empirical survey of the effectiveness of debiasing techniques for pre-trained language models},
  author={Meade, Nicholas and Poole-Dayan, Elinor and Reddy, Siva},
  journal={arXiv preprint arXiv:2110.08527},
  year={2021}
}

