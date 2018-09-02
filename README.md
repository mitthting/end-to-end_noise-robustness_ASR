# end-to-end noise-robustness ASR
This project provides speech denoise and speech recognition.

System required :
1. Python 2.7.x
2. Tensorflow 1.6.0
3. Keras 2.1.5
4. HTK 3.4.1 , if need to use GMM-HMM as recognizer
5. Matlab 2015 or later version , if need to use HTK

The speech feature contains [MFCC](https://drive.google.com/open?id=1Bju5ooZEwJO5eXi_18VSB2QkY-phimTj) and [AFE](https://drive.google.com/open?id=1gI1_6i9v2oH24vln4lCIoVW2opbQdFzo) features. All features are compose of 12 cepstrum and log-energy coefficient plus their corresponding first and second derivative (delta and acceleration) components.

The files or folder are explained as following. 
1. 'DAE_AFE_lossweight.py' : denoise speech feature
2. 'npy-matlab-master' : transform .npy format to HTK file format 
3. 'CTC' : build ene-to-end speech recognition via CTC. This work refers to [zzw922cn](https://github.com/zzw922cn/Automatic_Speech_Recognition)
4. 'CTC WER' : calculate CTC word error rate based on the HTK scripts setting
5. 'RECOGNIZER' : run GMM-HMM through HTK

The running step is described as follows if using different speech recognizer :

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

3. 'RECOGNIZER / train_recog_multi_etsi2' or 'RECOGNIZER / train_recog_clean_etsi2'
