Code for paper 

"Deep evidential fusion with uncertainty quantification and contextual discounting for multimodal medical image segmentation"


We have proposed a deep decision-level fusion architecture for multi-modality medical image segmentation. In this approach, features are first extracted from each modality using a deep neural network such as UNet. An evidence-mapping module based on prototypes in feature space then computes a Dempster-Shafer mass function at each voxel. To account for the varying reliability of different information sources in different contexts, the mass functions are transformed using the contextual discounting operation before being combined by Dempster's rule. The whole framework is trained end-to-end by minimizing a loss function that quantifies prediction error both at the modality level and after fusion.

Here we show the example usage to train BraTs2021 dataset when using nnFormer as the baseline feature extractor.

########Environment requirement######
Before using the code, please install the required packages using
```bash
 pip install -r requirements.txt
```

######### Data  ########
```bash
Put your own data on  ./dataset
```

######### Model  ########

The proposed models are put in networks/nnFormer
```bash
nnFormer_s_ds_flair.py
nnFormer_s_ds_t1ce.py
nnFormer_s_ds_t1.py
nnFormer_s_ds_t2.py
nnFormer_discounting.py
```


######### pre-trained weights #####

Download pre-trained weights of the feature extraction (FE) module flair, t1, t1Gd, and t2 using the link https://drive.google.com/drive/folders/1_iMHbOB2TQSeeoqtvjGZXL4hU6cdlarM?usp=drive_link

Put the downloaded FE weight in ./Pretrained_model.
If you want to use other SOTA FE modules, you can train your own FE on the single medical modality and put the trained model here. 

####### Training #######
and run
```bash
python script-TRAINING_nnFormer_discounting-early-stooping.py
```

#######   Test  ##########
Download the trained MMEF-nnFormer from link https://drive.google.com/drive/folders/165T-RceXnNCw7UhuZrxN3UCysV-BMORu?usp=drive_link
and run
```bash
python script-TRAINING_nnFormer_discounting-test.py
```

######### Citing this paper ########
```bash
@article{huang2025deep,
  title={Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation},
  author={Huang, Ling and Ruan, Su and Decazes, Pierre and Den{\oe}ux, Thierry},
  journal={Information Fusion},
  volume={113},
  pages={102648},
  year={2025},
  publisher={Elsevier}
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

