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
#for i, data in enumerate(uniqueText) :
#   num_to_text[i] = data

# 숫자화된 텍스트를 집어넣을거임 
numList = []
for i in text :
    numList.append(text_to_num[i])
    
firstInput = numList[117:117+25]
firstInput = tf.one_hot(firstInput,31)
firstInput = tf.expand_dims(firstInput, axis=0) #axis = 0
predictValue = model.predict(firstInput)
      
print(predictValue)
print(np.argmax(predictValue[0]))
print(numList[117+25])
