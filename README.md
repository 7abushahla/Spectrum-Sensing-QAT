# Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach
_Hamza A. Abushahla, Dara Varam, and Dr. Mohamed I. AlHajri_

This repository contains code and resources for the paper: "[Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234)"

## Introduction

## Datasets
We evaluate our models using the publicly available SDR and LTE datasets obtained from: https://github.com/wineslab/deepsense-spectrum-sensing-datasets

For SDR, in bin2hdf5.py, we set nsamples_per_file to 50000 (to match reported # of occurnces per channel) and buf to 32 and 128 (control window size). we keep test_size = 0.1 (to get 90% training % validation) and 10% testing. we keep stride = 12 (overlap between I/Q samples)

For LTE, in generateLTEDataset.m, we set niq to 32 and 128 (control window size) and varry snr_db between -20db and 20db. we keep Cross validation (train: 90%, test: 10%) of the generated data. rest of simulation stettings remain as provided by authors.

# Hardware Evaluation

## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```
@Article{abushahla2025cognitive,
AUTHOR = {Abushahla, Hamza A. and Varam, Dara and AlHajri, Mohamed I.},
TITLE = {Cognitive Radio Spectrum Sensing on the Edge: A Quantization-Aware Deep Learning Approach},
JOURNAL = { },
VOLUME = {},
YEAR = {},
NUMBER = {},
ARTICLE-NUMBER = {},
URL = {},
ISSN = {},
ABSTRACT = {},
DOI = {}
}
```

You can also reach out through email to: 
- Hamza Abushahla - b00090279@alumni.aus.edu
- Dara Varam - b00081313@alumni.aus.edu
- Dr. Mohamed AlHajri - mialhajri@aus.edu
