Code for paper 
```bash
"Deep evidential fusion with uncertainty quantification and contextual discounting for multimodal medical image segmentation"
```

We propose a new deep framework allowing us to merge multi-MRI image segmentation results using the formalism of Dempster-Shafer theory while taking into account the reliability of different modalities relative to different classes.

###Environment requirement###:
```bash
Before using the code, please install the required packages using pip install -r requirements.txt
```

###Data#####
```bash
Put your own data on  ./dataset
```

###Pretrained weights #####
```bash
Pre-Trained weights of FE module for flair, t1, t1Gd and t2 are located in ./Pretrained_model.
```

###Training #####
```bash
python script-TRAINING_nnFormer_discounting-early-stooping.py
```

###Test #####
```bash
python script-TRAINING_nnFormer_discounting-test.py
```

###########Citing this paper #############
```bash
@article{huang2023deep,
  title={Deep evidential fusion with uncertainty quantification and contextual discounting for multimodal medical image segmentation},
  author={Huang, Ling and Ruan, Su and Decazes, Pierre and Denoeux, Thierry},
  journal={arXiv preprint arXiv:2309.05919},
  year={2023}
}
@inproceedings{huang2022evidence,
  title={Evidence fusion with contextual discounting for multi-modality medical image segmentation},
  author={Huang, Ling and Denoeux, Thierry and Vera, Pierre and Ruan, Su},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={401--411},
  year={2022},
  organization={Springer}

}
```

