# PFLD implementation with tensorflow
It is an open surce program reference to https://arxiv.org/pdf/1902.10859.pdf , if you find any bugs or anything incorrect,you can notice it in the issues and pull request,be glad to receive you advices.     
And thanks @lucknote for helping fixing existing bugs.

## Datasets

**WFLW Dataset**

​    [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) contains 10000 faces (7500 for training and 2500 for testing)  with 98 fully manual annotated landmarks.

1. Training and Testing images [[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)]
2. WFLW  [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)

## Training & Testing

training :
~~~shell
$ python data/SetPreparation.py
$ train.sh
~~~

use tensorboard, open a new terminal
~~~
$ tensorboard  --logdir=./checkpoint/tensorboard/
~~~

testing:
~~~shell
$ python test.py
~~~
  
## Results:  
Sample images:  
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/10.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/121.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/17.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/19.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/21.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/52.jpg)
 ![Image text](https://github.com/guoqiangqi/PFLD/blob/master/data/sample_imgs/7.jpg)
        
 Sample gif:  

 ![Image text](data/sample_imgs/ucgif_20190809185908.gif)
 

## Bug fix
  1. The code for cauculating euler angles prediction loss have been added.  
  2. Fixed the memory leak bug:  
  The code has a flaw that i calculate euler angles ground-truth while training process,so the training speed have slowed down because  some work have to be finished on the cpu ,you should calculate the euler angles in the preprocess code    

## CONTACT US:

If you have any questiones ,please contact us! Also join our QQ group(945933636) for more information.

The document generated with DeepWiki was hosted on https://deepwiki.com/guoqiangqi/PFLD 
