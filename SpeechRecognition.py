# 導入函式庫
from preprocess import *
# import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,LSTM,ConvLSTM2D
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_test_org = y_test
# 類別變數轉為one-hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print("X_train.shape=", X_train.shape)

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
# 建立池化層，池化大小=2x2，取最大值
model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
# Add output layer
model.add(Dense(3, activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
earlystop = EarlyStopping(monitor="val_loss",
                            patience=50,
                            verbose=1,
                          restore_best_weights=True)

model.compile(loss='categorical_crossentropy',
              # optimizer=keras.optimizers.Adadelta(),
              optimizer='adam',
              metrics=['accuracy'])





# 進行訓練, 訓練過程會存在 train_history 變數中
model.fit(X_train, y_train_hot, batch_size=64, epochs=300, verbose=1, validation_data=(X_test, y_test_hot),callbacks=[earlystop])


X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
score = model.evaluate(X_test, y_test_hot, verbose=1)

# 模型存檔
# from keras.models import load_model
# model.save('ASR.h5')  # creates a HDF5 file 'model.h5'
# model.summary()

# 預測(prediction)
# mfcc = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
# mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
# print("labels=", get_labels())
# print("predict=", np.argmax(model.predict(mfcc_reshaped)))

import pandas as pd
# predictions = model.predict_classes(X_test)
predictions = np.argmax(model.predict(X_test), axis=-1)
print(pd.crosstab(y_test_org, predictions, rownames=['實際值'], colnames=['預測值']))

#
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


from sklearn.metrics import confusion_matrix
classes = ['Glass','Crash','Scream']
y_true = y_test_org
y_pred = predictions
cm = confusion_matrix(y_true, y_pred)
cm = (cm / 24)*100
plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
