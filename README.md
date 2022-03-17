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

一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一一

參與計畫:科技部補助研究計畫

計畫名稱:應用智慧機器人技術於商旅與高齡對象之接待與照護

執行起迄：2020/11/01~2021/10/31

計畫概述：	智慧機器人的應用越來越廣，漸漸與我們的日常生活息息相關。本計畫欲應用智慧機器人科技於商旅業及機構式高齡照護園區中訪客及住戶的接待與照護。本計畫訂定了兩大關鍵技術 (智慧視覺與智慧語音互動)及九大功能，例如：機器人逐地介紹館內設施、機器人主動打招呼、主動跟客人聊話題、機器人講笑話、用螢幕跟客人玩遊戲、巡邏館區及偵測意外事件等等。本產學計畫可以讓機器人進一步進化其智慧技術、讓商旅及機構式高齡照護達於科技智慧化、也可以讓智慧機器人應用進一步深入我們的生活中。
