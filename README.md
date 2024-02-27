# USSD-depression
Unsupervised Speaker Disentanglement for Depression Detection from Speech

Implementation and best models for the paper "A Privacy-preserving Unsupervised Speaker Disentanglement Method for Depression Detection from Speech" accepted at AAAI-ML4CMH Workshop. 

The Depression prediction code is originally based on DepAudioNet by Ma, et al. [5] and adapted from Andrew Bailey's GitHub repo  - https://github.com/adbailey1/DepAudioNet_reproduction [6]

To reproduce the best results using ComparE16 features and the LSTM-Only model along with a fusion of the Word2Vec model from our paper follow the following steps: 

1. Audio Model Data Preparation - 
    a. Download Daic-Woz Data. 
    b. Extract ComparE16 features only for the participant audio using the generate_features.sh script in the feature_extraction folder. Prerquistites - openSmile (https://github.com/audeering/opensmile)
    c. We use the DepAudioNet framework for data prep. cd to daic_woz_data_prep folder
    d. Change appropriate paths to the dataset in the config.py file. 
    e. Run the command `python-m run` which will prepare the ComparE16 features 

2. Audio Model Evaluation
    a. Download the model from https://drive.google.com/drive/folders/1nWSMS9NVmN4qmtug9WxmGYHevjaGeRSH?usp=sharing and save the model in the folder where ComparE16 features are created ( should be something like /Dataset/daic-woz-old/audio_feats/feats_DepAudioNet/compare16_delta_snv_exp)
    b. Set paths in the config file - config_compare16_cnn_lstm.py to match your paths. 
    c. Run the majority voting test command to get predictions for audio modality using ComparE16 - `python3 main_total_fscore_dist.py test --validate --prediction_metric=2 --threshold=total`

5. Text Model Data Preparation
    a. Repeat the same Word2Vec data prep using the Daic-WOZ DepAudioNet framework.  

6. Text Model Evaluation
    a. Same as the Audio model, run the test command for the Word2vecmodel. The best model is saved in the same drive folder here - https://drive.google.com/drive/folders/1nWSMS9NVmN4qmtug9WxmGYHevjaGeRSH?usp=sharing

7. Audio and Text Model Fusion
    a. Fusion of Word2Vec and Compare16 to reproduce the best result can be done using the following script - compare16_word2vec_fusion/text_fusion.ipynb



Other relevant repositories that may be useful to understand this repository: 

1. https://github.com/adbailey1/daic_woz_process
2. https://github.com/adbailey1/DepAudioNet_reproduction
3. https://github.com/kingformatty/NUSD


References

If you found this repo useful, please consider citing our work -  

```
[1] Ravi, V., Wang, J., Flint, J., Alwan, A. (2022) A Step Towards Preserving Speakers’ Identity While Detecting Depression Via Speaker Disentanglement. Proc. Interspeech 2022, 3338-3342, doi: 10.21437/Interspeech.2022-10798

@inproceedings{ravi22_interspeech,
  author={Vijay Ravi and Jinhan Wang and Jonathan Flint and Abeer Alwan},
  title={{A Step Towards Preserving Speakers’ Identity While Detecting Depression Via Speaker Disentanglement}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={3338--3342},
  doi={10.21437/Interspeech.2022-10798}
}
```

```
[2] Wang, J., Ravi, V., Alwan, A. (2023) Non-uniform Speaker Disentanglement For Depression Detection From Raw Speech Signals. Proc. INTERSPEECH 2023, 2343-2347, doi: 10.21437/Interspeech.2023-2101

@inproceedings{wang23pa_interspeech,
  author={Jinhan Wang and Vijay Ravi and Abeer Alwan},
  title={{Non-uniform Speaker Disentanglement For Depression Detection From Raw Speech Signals}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2343--2347},
  doi={10.21437/Interspeech.2023-2101}
}
```

```
[3] Vijay Ravi, Jinhan Wang, Jonathan Flint, Abeer Alwan, Enhancing accuracy and privacy in speech-based depression detection through speaker disentanglement, Computer Speech & Language, Volume 86, 2024, 101605, ISSN 0885-2308, https://doi.org/10.1016/j.csl.2023.101605.

@article{RAVI2024101605,
title = {Enhancing accuracy and privacy in speech-based depression detection through speaker disentanglement},
journal = {Computer Speech & Language},
volume = {86},
pages = {101605},
year = {2024},
issn = {0885-2308},
doi = {https://doi.org/10.1016/j.csl.2023.101605},
url = {https://www.sciencedirect.com/science/article/pii/S0885230823001249},
author = {Vijay Ravi and Jinhan Wang and Jonathan Flint and Abeer Alwan},
keywords = {Depression-detection, Speaker-disentanglement, Privacy}}
```

```
[4] Ravi, V., Wang, J., Flint, J., Alwan, A. (2024) A Privacy-Preserving Unsupervised Speaker Disentanglement Method for Depression Detection from Speech

@inproceedings{ravi24_ml4cmh_aaai,
  author={Vijay Ravi and Jinhan Wang and Jonathan Flint and Abeer Alwan},
  title={{A Privacy-Preserving Unsupervised Speaker Disentanglement Method for Depression Detection from Speech}},
  year=2024,
  booktitle={Ml4CMH Workshop, AAAI, 2024}
}
```

other relevant work - 

```
[5] Xingchen Ma, Hongyu Yang, Qiang Chen, Di Huang, and Yunhong Wang. 2016. DepAudioNet: An Efficient Deep Model for Audio based Depression Classification. In Proceedings of the 6th International Workshop on Audio/Visual Emotion Challenge (AVEC '16). Association for Computing Machinery, New York, NY, USA, 35–42. https://doi.org/10.1145/2988257.2988267
```
```
[6] Bailey, A., & Plumbley, M. D. (2021, August). Gender bias in depression detection using audio features. In 2021 29th European Signal Processing Conference (EUSIPCO) (pp. 596-600). IEEE.
```

