# Fair and private language models

This repository contains the code for my master thesis entitled "Trade-Offs Between Privacy and Fairness in Language Models", written at the Technical University of Darmstadt.
The aim of the work was to investigate the influence of debiasing on leakage and the influence of differential privacy on gender bias in GPT-2.
The repository is divided as follows:

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

