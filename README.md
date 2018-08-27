# end-to-end noise-robustness ASR
This project provides speech denoise and speech recognition.

System required :
1. Python 2.7 or later version
2. HTK 3.4.1 , if need to use GMM-HMM as recognizer
3. Matlab 2014 or later version , if need to use HTK

The required dataset will be uploaded as soon as possible.

The files or folder are explained as following. 
1. 'DAE_AFE_lossweight.py' : denoise speech feature
2. 'npy-matlab-master' : transform .npy format to HTK file fomat 
3. 'mvmulti' : classify multi train data into 20 subsets, executed if need to use HTK
4. 'CTC' : build ene-to-end speech recognition via CTC. This work refers to [zzw922cn](https://github.com/zzw922cn/Automatic_Speech_Recognition) and simply modify
5. 'CTC WER' : calculate CTC word error rate based on the HTK scripts setting
6. 'RECOGNIZER' : run GMM-HMM through HTK

The running step is described as follows if using different speech recognizer

# denoise + end-to-end
Execution Steps : 
1. 'DAE_AFE_lossweight.py'

2. 'CTC / main / train.py'

3. 'CTC / main / test.py'

4. 'CTC WER / RECOG_TESTx' or 'CTC WER / test.sh'


# denoise + GMM-HMM
Execution Steps : 
1. 'DAE_AFE_lossweight.py'

2. 'npy-matlab-master / NPY2HTK.m'

3. 'mvmulti.sh'

4. 'RECOGNIZER / train_recog_multi_etsi2' or 'RECOGNIZER / train_recog_clean_etsi2'
