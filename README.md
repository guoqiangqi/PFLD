# PFLD
Table 1:  
  The code for cauculating euler angles prediction loss have been released.

Table 2:     
  I`ve done the job and fixed the memory leak bug:
  The code has a flaw that i calculate euler angles ground-truth while training process,so the training speed have slowed down because  some work have to be finished on the cpu ,you should calculate the euler angles in the preprocess code    
      
Table 3：It is an open surce program reference to https://arxiv.org/pdf/1902.10859.pdf , if you find any bugs or anything incorrect,you can notice it in the issues and pull request,be glad to receive you advices.     
And thanks @lucknote for helping fixing existing bugs.
  
EASY TO TRAIN:
>STEP1:data/SetPreparation.py  
>STEP2:train.sh
  
  
SAMPLE IMGS:  

 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/10.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/121.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/17.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/19.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/21.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/52.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/7.jpg)
        
 SAMPLE VIDEO:  

 ![Image text](data/sample_imgs/ucgif_20190809185908.gif)
