# end-to-end noise-robustness ASR
This project provides speech denoise and speech recognition.

System required :
1. Python 2.7 or later version
2. HTK 3.4.1 , if need to use GMM-HMM as recognizer
3. Matlab 2014 or later version , if need to use HTK

The required dataset will be upload as soon as possble.

The files or folder are explained as following. 
1. 'DAE_AFE_lossweight' : denoise speech feature
2. 'npy-matlab-master' : transform .npy format to HTK file fomat 
3. 'mvmulti' : classify multi train data into 20 subsets, executed if need to use HTK
4. 'CTC' : build ene-to-end speech recognition via CTC. This work is refered to [zzw922cn](https://github.com/zzw922cn/Automatic_Speech_Recognition) and simply modified
5. 'CTC WER' : calculate CTC word error rate based on the HTK scripts setting
6. 'RECOGNIZER' : run GMM-HMM through HTK
