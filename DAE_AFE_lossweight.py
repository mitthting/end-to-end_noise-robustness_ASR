
import os, sys
import numpy as np

print 'Read Multi Data Start ...'

current_path = '/home/ding/Forgithub/'

baseDir = current_path

trainingSet = 'multi/'

subDir = [ 'clean1' , 'clean2' , 'clean3' , 'clean4' , 
           'N1_SNR5' , 'N2_SNR5' , 'N3_SNR5' , 'N4_SNR5',
           'N1_SNR10', 'N2_SNR10', 'N3_SNR10', 'N4_SNR10',
           'N1_SNR15', 'N2_SNR15', 'N3_SNR15', 'N4_SNR15',
           'N1_SNR20', 'N2_SNR20', 'N3_SNR20', 'N4_SNR20']


training_list = []

multiTR_path = []

num_frame_group = 11

multi_length = []

multi_clean = []
multi_SNR20 = []
multi_SNR15 = []
multi_SNR10 = []
multi_SNR5 = []


for h in range(len(subDir)):
    dirs = os.listdir(baseDir + trainingSet + subDir[h])

    for num in range(len(dirs)):
		
        tmp = np.loadtxt(baseDir + trainingSet + subDir[h] + '/' + dirs[num])
	   
        multi_length.append(tmp.size/39)
        
        multiTR_path.append(baseDir + trainingSet + '/' + dirs[num])

        zero = np.zeros(((num_frame_group -1)/2,39))
        tmp = np.concatenate((zero, tmp), axis=0)
        tmp = np.concatenate((tmp, zero), axis=0)

		# CMVN
        tmp = np.transpose(tmp)
	
        cur_mean = np.mean(tmp, axis=1)
        cur_std  = np.std(tmp, axis=1, ddof=1)
        tmp = tmp - cur_mean[:, None]
        tmp = tmp / cur_std[:, None]
        tmp = np.nan_to_num(tmp)
	
        tmp = np.transpose(tmp)
		
        for num_split_group in range(tmp.size/39-num_frame_group+1): 
			
			temp_data = np.concatenate((tmp[num_split_group], tmp[num_split_group+1], tmp[num_split_group+2], tmp[num_split_group+3], tmp[num_split_group+4], tmp[num_split_group+5], tmp[num_split_group+6], tmp[num_split_group+7], tmp[num_split_group+8], tmp[num_split_group+9], tmp[num_split_group+10]), axis=0)
			training_list.append(temp_data)
			if h<4 :
			    multi_clean.append(temp_data)
			elif h>=4 and h<8 :
			    multi_SNR5.append(temp_data)
			elif h>=8 and h<12 :
			    multi_SNR10.append(temp_data)
			elif h>=12 and h<16 :
			    multi_SNR15.append(temp_data)
			elif h>=16 and h<20 :
			    multi_SNR20.append(temp_data)

x_train_multi = np.array(training_list)

multi_clean = np.array(multi_clean)
multi_SNR5 = np.array(multi_SNR5)
multi_SNR10 = np.array(multi_SNR10)
multi_SNR15 = np.array(multi_SNR15)
multi_SNR20 = np.array(multi_SNR20)


print 'Read Multi Data End ...'

print 'Read Clean Data Start ...'

baseDir = current_path

trainingSet = 'clean/'

training_list = []

clean_clean = []
clean_SNR20 = []
clean_SNR15 = []
clean_SNR10 = []
clean_SNR5 = []

for h in range(len(subDir)):
    dirs = os.listdir(baseDir + trainingSet + subDir[h])

    for num in range(len(dirs)):
		
		tmp = np.loadtxt(baseDir + trainingSet + subDir[h] + '/' + dirs[num])
		    
		tmp = np.transpose(tmp)
	
		cur_mean = np.mean(tmp, axis=1)
		cur_std  = np.std(tmp, axis=1, ddof=1)
		tmp = tmp - cur_mean[:, None]
		tmp = tmp / cur_std[:, None]
		tmp = np.nan_to_num(tmp)
	
		tmp = np.transpose(tmp)
	
		for i in range(len(tmp)):
		    training_list.append(tmp[i])
		    if h<4 :
			    clean_clean.append(tmp[i])
		    elif h>=4 and h<8 :
			    clean_SNR5.append(tmp[i])
		    elif h>=8 and h<12 :
			    clean_SNR10.append(tmp[i])
		    elif h>=12 and h<16 :
			    clean_SNR15.append(tmp[i])
		    elif h>=16 and h<20 :
			    clean_SNR20.append(tmp[i])
          
x_train_clean = np.array(training_list)

clean_clean = np.array(clean_clean)
clean_SNR5 = np.array(clean_SNR5)
clean_SNR10 = np.array(clean_SNR10)
clean_SNR15 = np.array(clean_SNR15)
clean_SNR20 = np.array(clean_SNR20)

print 'Read Clean Data End ...'

print 'Start training ...'

from keras.layers import *
from keras.models import *

import keras.optimizers

cleanw = []
SNR5w = []
SNR10w = []
SNR15w = []
SNR20w = []

for i in range(273713) : #clean
    cleanw.append(1.0)
for i in range(276048) : #5dB
    SNR5w.append(1.0)
for i in range(272158) : #10dB
    SNR10w.append(1.0)
for i in range(271282) : #15dB
    SNR15w.append(1.0)
for i in range(270481) : #20dB
    SNR20w.append(1.0)

cleanw = np.array(cleanw)
SNR5w = np.array(SNR5w)
SNR10w = np.array(SNR10w)
SNR15w = np.array(SNR15w)
SNR20w = np.array(SNR20w)

weightloss = np.concatenate((cleanw,SNR5w,SNR10w,SNR15w,SNR20w) , axis=0)

leakyrelu = keras.layers.LeakyReLU(alpha=0.3)

# optimizer
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)

# instantiate model
model = Sequential()

model.add(Dense(500, input_dim=429))
model.add(leakyrelu)
model.add(Dense(500))
model.add(leakyrelu)
model.add(Dense(273))
model.add(leakyrelu)
model.add(Dense(300))
model.add(leakyrelu)
model.add(Dense(300))
model.add(leakyrelu)
model.add(Dense(39))

model.compile(loss='mse', optimizer=adam)

rclean=[]
rSNR5=[]
rSNR10=[]
rSNR15=[]
rSNR20=[]

rclean.append(1.0)
rSNR5.append(1.0)
rSNR10.append(1.0)
rSNR15.append(1.0)
rSNR20.append(1.0)

alphac = 0.1
alpha5 = 0.1
alpha10 = 0.1
alpha15 = 0.1
alpha20 = 0.1


for epoch in range(1) :

    model.fit(x_train_multi, x_train_clean, epochs=1, batch_size=100, shuffle=True, sample_weight=weightloss)

    loss_clean = model.evaluate(multi_clean, clean_clean, verbose=0, batch_size=500)
    loss_SNR5  = model.evaluate(multi_SNR5 , clean_SNR5 , verbose=0, batch_size=500)
    loss_SNR10 = model.evaluate(multi_SNR10, clean_SNR10, verbose=0, batch_size=500)
    loss_SNR15 = model.evaluate(multi_SNR15, clean_SNR15, verbose=0, batch_size=500)
    loss_SNR20 = model.evaluate(multi_SNR20, clean_SNR20, verbose=0, batch_size=500)
   
    avgloss = (loss_clean + loss_SNR5 + loss_SNR10 + loss_SNR15 + loss_SNR20)/5
    
    weightloss = []

    cleanw = cleanw + alphac*(loss_clean-avgloss)
    SNR5w  = SNR5w  + alpha5*(loss_SNR5-avgloss)
    SNR10w = SNR10w + alpha10*(loss_SNR10-avgloss)
    SNR15w = SNR15w + alpha15*(loss_SNR15-avgloss)
    SNR20w = SNR20w + alpha20*(loss_SNR20-avgloss)
    
    weightloss = np.concatenate((cleanw,SNR5w,SNR10w,SNR15w,SNR20w) , axis=0)

    alpha5  = alpha5  * 0.95
    alpha10 = alpha10 * 0.95
    alpha15 = alpha15 * 0.95
    alpha20 = alpha20 * 0.95
    alphac  = alphac  * 0.95

    rclean.append(cleanw[0])
    rSNR5.append(SNR5w[0])
    rSNR10.append(SNR10w[0])
    rSNR15.append(SNR15w[0])
    rSNR20.append(SNR20w[0])
    
    print 'Epoch = ', epoch+2

print 'training is over.'

rrclean = np.array(rclean)
rrSNR5 = np.array(rSNR5)
rrSNR10 = np.array(rSNR10)
rrSNR15 = np.array(rSNR15)
rrSNR20 = np.array(rSNR20)

np.savetxt('clean.txt', rrclean)
np.savetxt('SNR5.txt' , rrSNR5)
np.savetxt('SNR10.txt', rrSNR10)
np.savetxt('SNR15.txt', rrSNR15)
np.savetxt('SNR20.txt', rrSNR20)

print 'record is over'

cleanw = None
SNR5w = None
SNR10w = None
SNR15w = None
SNR20w = None

weightloss = None


print 'Create Directory ...'
output_dir = 'Output_DAE_AFE_lossweight1'

os.makedirs(current_path + output_dir)
os.makedirs(current_path + output_dir + '/multi')
os.makedirs(current_path + output_dir + '/testA')
os.makedirs(current_path + output_dir + '/testB')
os.makedirs(current_path + output_dir + '/testC')

os.makedirs(current_path + output_dir + '/testA/clean1')
os.makedirs(current_path + output_dir + '/testA/clean2')
os.makedirs(current_path + output_dir + '/testA/clean3')
os.makedirs(current_path + output_dir + '/testA/clean4')
os.makedirs(current_path + output_dir + '/testA/N1_SNR0')
os.makedirs(current_path + output_dir + '/testA/N1_SNR5')
os.makedirs(current_path + output_dir + '/testA/N1_SNR10')
os.makedirs(current_path + output_dir + '/testA/N1_SNR15')
os.makedirs(current_path + output_dir + '/testA/N1_SNR20')
os.makedirs(current_path + output_dir + '/testA/N1_SNR-5')
os.makedirs(current_path + output_dir + '/testA/N2_SNR0')
os.makedirs(current_path + output_dir + '/testA/N2_SNR5')
os.makedirs(current_path + output_dir + '/testA/N2_SNR10')
os.makedirs(current_path + output_dir + '/testA/N2_SNR15')
os.makedirs(current_path + output_dir + '/testA/N2_SNR20')
os.makedirs(current_path + output_dir + '/testA/N2_SNR-5')
os.makedirs(current_path + output_dir + '/testA/N3_SNR0')
os.makedirs(current_path + output_dir + '/testA/N3_SNR5')
os.makedirs(current_path + output_dir + '/testA/N3_SNR10')
os.makedirs(current_path + output_dir + '/testA/N3_SNR15')
os.makedirs(current_path + output_dir + '/testA/N3_SNR20')
os.makedirs(current_path + output_dir + '/testA/N3_SNR-5')
os.makedirs(current_path + output_dir + '/testA/N4_SNR0')
os.makedirs(current_path + output_dir + '/testA/N4_SNR5')
os.makedirs(current_path + output_dir + '/testA/N4_SNR10')
os.makedirs(current_path + output_dir + '/testA/N4_SNR15')
os.makedirs(current_path + output_dir + '/testA/N4_SNR20')
os.makedirs(current_path + output_dir + '/testA/N4_SNR-5')

os.makedirs(current_path + output_dir + '/testB/clean1')
os.makedirs(current_path + output_dir + '/testB/clean2')
os.makedirs(current_path + output_dir + '/testB/clean3')
os.makedirs(current_path + output_dir + '/testB/clean4')
os.makedirs(current_path + output_dir + '/testB/N1_SNR0')
os.makedirs(current_path + output_dir + '/testB/N1_SNR5')
os.makedirs(current_path + output_dir + '/testB/N1_SNR10')
os.makedirs(current_path + output_dir + '/testB/N1_SNR15')
os.makedirs(current_path + output_dir + '/testB/N1_SNR20')
os.makedirs(current_path + output_dir + '/testB/N1_SNR-5')
os.makedirs(current_path + output_dir + '/testB/N2_SNR0')
os.makedirs(current_path + output_dir + '/testB/N2_SNR5')
os.makedirs(current_path + output_dir + '/testB/N2_SNR10')
os.makedirs(current_path + output_dir + '/testB/N2_SNR15')
os.makedirs(current_path + output_dir + '/testB/N2_SNR20')
os.makedirs(current_path + output_dir + '/testB/N2_SNR-5')
os.makedirs(current_path + output_dir + '/testB/N3_SNR0')
os.makedirs(current_path + output_dir + '/testB/N3_SNR5')
os.makedirs(current_path + output_dir + '/testB/N3_SNR10')
os.makedirs(current_path + output_dir + '/testB/N3_SNR15')
os.makedirs(current_path + output_dir + '/testB/N3_SNR20')
os.makedirs(current_path + output_dir + '/testB/N3_SNR-5')
os.makedirs(current_path + output_dir + '/testB/N4_SNR0')
os.makedirs(current_path + output_dir + '/testB/N4_SNR5')
os.makedirs(current_path + output_dir + '/testB/N4_SNR10')
os.makedirs(current_path + output_dir + '/testB/N4_SNR15')
os.makedirs(current_path + output_dir + '/testB/N4_SNR20')
os.makedirs(current_path + output_dir + '/testB/N4_SNR-5')

os.makedirs(current_path + output_dir + '/testC/clean1')
os.makedirs(current_path + output_dir + '/testC/clean2')
os.makedirs(current_path + output_dir + '/testC/N1_SNR0')
os.makedirs(current_path + output_dir + '/testC/N1_SNR5')
os.makedirs(current_path + output_dir + '/testC/N1_SNR10')
os.makedirs(current_path + output_dir + '/testC/N1_SNR15')
os.makedirs(current_path + output_dir + '/testC/N1_SNR20')
os.makedirs(current_path + output_dir + '/testC/N1_SNR-5')
os.makedirs(current_path + output_dir + '/testC/N2_SNR0')
os.makedirs(current_path + output_dir + '/testC/N2_SNR5')
os.makedirs(current_path + output_dir + '/testC/N2_SNR10')
os.makedirs(current_path + output_dir + '/testC/N2_SNR15')
os.makedirs(current_path + output_dir + '/testC/N2_SNR20')
os.makedirs(current_path + output_dir + '/testC/N2_SNR-5')


print 'Start testing ...'

print 'Multi train data put into model...'

x_train_multi_new = model.predict(x_train_multi, batch_size=500)

print 'End Feeding ...'

print 'Write multi train data into directory ...'


index_count = 0

temp = []

for file_num in range(len(multi_length)): 
    for frame_num in range(multi_length[file_num]): 
        temp = np.concatenate((temp, x_train_multi_new[index_count + frame_num]), axis=0) 
    
    temp = temp.reshape(multi_length[file_num],39)
    
    temp = np.transpose(temp)
    savepath = multiTR_path[file_num].replace('multi',output_dir+'/multi')
    savefile = savepath.split('.')
    np.save(savefile[0], temp)
    
    index_count = index_count + multi_length[file_num]
        
    temp=[]


for testindex in ['testA/', 'testB/', 'testC/'] :
	
	baseDir = current_path
	trainingSet = testindex
	if testindex != 'testC/' :
		subDir = ['clean1', 'N1_SNR0', 'N1_SNR5', 'N1_SNR10', 'N1_SNR15', 'N1_SNR20', 'N1_SNR-5',
		'clean2', 'N2_SNR0', 'N2_SNR5', 'N2_SNR10', 'N2_SNR15', 'N2_SNR20', 'N2_SNR-5',
		'clean3', 'N3_SNR0', 'N3_SNR5', 'N3_SNR10', 'N3_SNR15', 'N3_SNR20', 'N3_SNR-5',
		'clean4', 'N4_SNR0', 'N4_SNR5', 'N4_SNR10', 'N4_SNR15', 'N4_SNR20', 'N4_SNR-5',]
	else :
		subDir = ['clean1', 'N1_SNR0', 'N1_SNR5', 'N1_SNR10', 'N1_SNR15', 'N1_SNR20', 'N1_SNR-5',
		'clean2', 'N2_SNR0', 'N2_SNR5', 'N2_SNR10', 'N2_SNR15', 'N2_SNR20', 'N2_SNR-5']

	set_list = []

	set_path = []

	num_frame_group = 11

	set_length = []

	for h in range(len(subDir)):
		dirs = os.listdir(baseDir + trainingSet + subDir[h])
		
		for num in range(len(dirs)):
		    tmp = np.loadtxt(baseDir + trainingSet + subDir[h] + '/' + dirs[num])

		    set_length.append(tmp.size/39)
		    set_path.append(baseDir + trainingSet + subDir[h] +'/' + dirs[num])

		    zero = np.zeros(((num_frame_group -1)/2,39))
		    tmp = np.concatenate((zero, tmp), axis=0)
		    tmp = np.concatenate((tmp, zero), axis=0)

		    tmp = np.transpose(tmp)
		
		    cur_mean = np.mean(tmp, axis=1)
		    cur_std  = np.std(tmp, axis=1, ddof=1)
		    tmp = tmp - cur_mean[:, None]
		    tmp = tmp / cur_std[:, None]
		    tmp = np.nan_to_num(tmp)
		
		    tmp = np.transpose(tmp)

		    
		    for num_split_group in range(tmp.size/39-num_frame_group+1):
		        
		        temp_data = np.concatenate((tmp[num_split_group], tmp[num_split_group+1], tmp[num_split_group+2], tmp[num_split_group+3], tmp[num_split_group+4], tmp[num_split_group+5], tmp[num_split_group+6], tmp[num_split_group+7], tmp[num_split_group+8], tmp[num_split_group+9], tmp[num_split_group+10]), axis=0)
		        set_list.append(temp_data)
		
		x_test = np.array(set_list)
		
		x_test_new = model.predict(x_test, batch_size=500)
		
		index_count = 0

		temp = []

		for file_num in range(len(set_length)):
		    for frame_num in range(set_length[file_num]):
		        temp = np.concatenate((temp, x_test_new[index_count + frame_num]), axis=0)
		    
		    temp = temp.reshape(set_length[file_num],39)
		
		    temp = np.transpose(temp)
		    savepath = set_path[file_num].replace(testindex,output_dir+'/'+testindex)
		    savefile = savepath.split('.')
		    np.save(savefile[0], temp)
		    
		    index_count = index_count + set_length[file_num]
		    
		    temp=[]

		set_list = []
		set_path = []
		set_length = []


