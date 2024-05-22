import numpy as np
import tensorflow as tf

text = open('./pianoabc.txt','r').read()

# Bag of words ( 유니크한 단어 주머니 ) 이거 가지고 넘버링 하면 됨
uniqueText = list(set(text))
uniqueText.sort()

text_to_num = {}
num_to_text = {}

# 문자들을 숫자로 치환
for i, data in enumerate(uniqueText) :
    text_to_num[data] = i # text_to_num 의 "특정 문자가 " 숫자로 바뀌는거임 

# 숫자들을 문자로 치환
#for i, data in enumerate(uniqueText) :
#   num_to_text[i] = data


# 숫자화된 텍스트를 집어넣을거임 
numList = []
for i in text :
    numList.append(text_to_num[i])

# 학습 데이터
X = [] 

# 정답 데이터
Y = []

for i in range( 0, len(numList) -25 ):
    X.append(numList[ i : i + 25 ])
    Y.append(numList[ i + 25 ])


# 원핫인코딩
X = tf.one_hot( X, 31 )
Y = tf.one_hot( Y, 31 )

print(X)
print(Y)

# np.array(X).shape

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM( 100, input_shape=( 25 , 31 )),
    tf.keras.layers.Dense( 31 , activation="softmax" )
])
model.summary()
model.compile( loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )
model.fit( X, Y, batch_size=64, epochs=1, verbose=2 )
model.save("./rnn_model_0")

# input_1 = tf.keras.layers.Input(shape=[25,31])
# lstm_1 = tf.keras.layers.LSTM(100,return_sequences=True)(input_1)
# lstm_2 = tf.keras.layers.LSTM(64,return_sequences=True)(lstm_1)
# lstm_3 = tf.keras.layers.LSTM(32)(lstm_2)

# output = tf.keras.layers.Dense(31, activation="softmax")(lstm_3)

# functionalModel = tf.keras.Model(input_1, output)
# functionalModel.compile( loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )
# functionalModel.fit( X, Y, batch_size=64, epochs=1, verbose=2 )
# functionalModel.save("./rnn_model_1")