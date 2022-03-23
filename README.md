# Deep-Learning-Abnormal-Sound-Recognition 特殊音辨識(玻璃破碎聲、碰撞聲、尖叫聲)

深度學習-特殊音辨識(玻璃破碎聲、碰撞聲、尖叫聲)

這篇文章為使用Tensorflow的Keras進行特殊音辨識

需要先用pip install or conda install 安裝環境套件

我的tensorflow是2.2.0版本 python是3.8

然後下載dataset放到正確的路徑即可

這邊放上dataset的下載連結 https://drive.google.com/drive/folders/1-NwabwBIeS9saHYmmlo-Ik3aYzKQ52Ig?usp=sharing

我們萃取特徵的方式是使用MFCC，如果再加上其他特徵進行模型訓練的話，效果一定會更好

環境安裝好後即可python SpeechRecognition.py執行程式

preprocess.py是前處理的程式，主要是在做切音框與特徵萃取的事情

但是我在SpeechRecognition.py有把它呼叫進來，所以直接執行SpeechRecognition.py就可以了。

訓練流程:

1.Dataset & Label : CHEN, Oscal T.-C., et al. Voice-activity home care system. In: 2016 IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI). IEEE, 2016. p. 110-113.

2.特徵萃取 : MFCC

3.特徵選擇 : 卷積神經網路(CNN)

4.分類器 : 3-Layers Fully Connect
