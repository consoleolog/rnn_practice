import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./model')

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
for i, data in enumerate(uniqueText) :
  num_to_text[i] = data

# 숫자화된 텍스트를 집어넣을거임 
numList = []
for i in text :
    numList.append(text_to_num[i])


music = []
첫입력값 = numList[117:117+25]
첫입력값 = tf.one_hot(첫입력값,31)
첫입력값 = tf.expand_dims(첫입력값,axis=0)

for i in range(200):

    # predict로 다음문자 예측
    예측값 = model.predict(첫입력값)
    예측값 = np.argmax(예측값[0])
    
    # 예측한 다음문자 [] 저장하기
    music.append(예측값)
    # 첫 입력값 앞에 짜르기
    다음입력값 = 첫입력값.numpy()[0][1:]
    
    # 원핫인코딩하기, expand dims 하기
    one_hot_num = tf.one_hot(예측값,31)
    첫입력값 = np.vstack([다음입력값, one_hot_num.numpy()])
    첫입력값 = tf.expand_dims(첫입력값, axis=0)

music_text = []

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))










