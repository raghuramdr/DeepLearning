import os
import glob as glob

os.chdir('/home/raghuram/Desktop/Computer Vision Projects/101_ObjectCategories')
dir_list = os.listdir(os.getcwd())
cwd = os.getcwd()
label_ctr = 0
pathname = '/home/raghuram/caffe/examples/images/101_ObjectCategories/'
training = open('training.txt','a')
testing = open('testing.txt','a')

for subdir in dir_list:
    if subdir == 'training.txt' or subdir == 'testing.txt':
       continue
    else:
        os.chdir(cwd+'/'+subdir)
        file_list = glob.glob('*.jpg')
        for i in xrange(len(file_list)):
            filename = file_list[i]
            if i<=int(0.7*len(file_list)):
               info = pathname+subdir+'/'+filename
               training.write(info+" "+str(label_ctr)+"\n")
            else:
               info = pathname+subdir+'/'+filename
               testing.write(info+" "+str(label_ctr)+"\n")
        label_ctr+=1

testing.close()
training.close()
    
           
