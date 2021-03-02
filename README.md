# PFLD implementation with tensorflow
It is an open surce program reference to https://arxiv.org/pdf/1902.10859.pdf , if you find any bugs or anything incorrect,you can notice it in the issues and pull request,be glad to receive you advices.     
And thanks @lucknote for helping fixing existing bugs.

## Easy To Train:
```
> python data/SetPreparation.py  
> train.sh
```

## Bug fix
  1. The code for cauculating euler angles prediction loss have been added.  
  2. Fixed the memory leak bug:  
  The code has a flaw that i calculate euler angles ground-truth while training process,so the training speed have slowed down because  some work have to be finished on the cpu ,you should calculate the euler angles in the preprocess code    
  
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
 
## CONTACT US:

If you have any questiones ,please contact us! Also join our QQ group(945933636) for more information.
