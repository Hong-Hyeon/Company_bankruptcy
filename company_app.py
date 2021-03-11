import pandas as pd
import numpy as np
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def data_prepro(data):
    df = pd.DataFrame()
    # ROA 체크
    # ROA는 가장 낮은 것으로. -> 총자산 수익률
    df['x1'] = data.iloc[:,1]
    # 영업이익
    df['x2'] = data.iloc[:,5]
    # 세후 순 이자율 - 순이익 / 순매출
    df['x3'] = data.iloc[:,8]
    # 현금 흐름 속도
    df['x4'] = data.iloc[:,13]
    # 이자부 부채금리
    df['x5'] = data.iloc[:,14]
    # 주당 현금흐름
    df['x6'] = data.iloc[:,20]
    # 당기순이익 증가율
    df['x7'] = data.iloc[:,27]
    # 순 가치 성장률
    df['x8'] = data.iloc[:,30]
    # 현금 재투자 비율
    df['x9'] = data.iloc[:,32]
    # 총 부채 비율
    df['x10'] = data.iloc[:,36]
    # 유동 자산 / 총 자산
    df['x11'] = data.iloc[:,56]
    # 현금 회전율
    df['x12'] = data.iloc[:,74]
    # 책임자산 플래그 - 총 부채가 총 자산을 초과하는 경우 1, 아닌 경우 0
    df['x13'] = data.iloc[:,85]
    # 라벨링 / 회사가 망했는지 안망했는지
    df['y'] = data.iloc[:,0]

    # Normalization
    scaler = MinMaxScaler()
    scale_cols = []
    for i in range(len(df.columns)):
        if i == 13:
            scale_cols.append('y')
            break
        scale_cols.append('x{}'.format(i+1))
    scaled = scaler.fit_transform(df[scale_cols])

    scaled_df = pd.DataFrame(scaled, columns=scale_cols)

    scaled_df.to_csv('./data/scaled_data.csv', index=False)

    return scaled_df

def modeling():
    model = Sequential([
        Dense(32, input_shape=[13], activation='relu'),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

if __name__ == '__main__':
    df = pd.read_csv('./data/data.csv')
    scaled_df = data_prepro(df)

    x = scaled_df.iloc[:,:-1]
    y = scaled_df.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    model = modeling()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    filepath = './data/checkpoint'
    modelpath = './data/models'

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    check = './data/checkpoint/check.ckpt'
    checkpoint = ModelCheckpoint(filepath=check,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1,
                                 monitor='val_loss')

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=20, callbacks=[checkpoint])

    model.load_weights(check)

    print(model.evaluate(x_test,y_test))

    model.save('./data/models/company.h5')

    # predict = model.predict([0.508360552,0.601716658,0.809361528,0.46702398,
    #                         6.363,0.324114027,0.48,4.83,
    #                         0.384234939,5.1,0.447942035,0.0734,0])
    # print(predict)
    # print(predict.shape)