import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from icecream import ic


class CabbageModel:
     def __init__(self) -> None:
          self.model = os.path.join(basedir, 'model')
          self.data = None
          self.x_data = None
          self.y_data = None
          
          
     
     def cabbage(self, avg_temp, min_temp, max_temp, rain_fall):
          print(f'훅에 전달된 avg_temp: {avg_temp}, min_temp: {min_temp}, max_temp: {max_temp}, rain_fall: {rain_fall}')
          tf.reset_default_graph()
          with tf.Session() as sess:
               sess.run(tf.global_variables_initializer())
               tf.train.import_meta_graph(self.model + '/cabbage_'+avg_temp+min_temp+max_temp+rain_fall+'/saved.cpkt.meta')
               graph = tf.get_default_graph()
               w1 = graph.get_tensor_by_name('w1:10')
               w2 = graph.get_tensor_by_name('w2:-5')
               w3 = graph.get_tensor_by_name('w3:20')
               w4 = graph.get_tensor_by_name('w4:0.8')
               feed_dict = {w1: float(avg_temp), w2: float(min_temp), w3: float(max_temp), w4: float(rain_fall)}
               op_to_restore = graph.get_tensor_by_name('op_'+avg_temp+min_temp+max_temp+rain_fall+':0')
               result =sess.run(op_to_restore, feed_dict)
               print(f'최종결과:{result}')
          return result
     
     #모델을 위한 전처리
     def preprocessing(self):
          self.data = pd.read_csv('C:/flask/model/cabbage/price_data.csv')
          
          # avgTemp,minTemp,maxTemp,rainFall,avgPrice
          xy = np.array(self.data, dtype=np.float32)
          #np.array를 사용하면 dataframe을 슬라이싱 인덱싱이 가능한 ndarray로 변환이 가능하다. 
          #( Dataframe 과 ndarray 모두 2차원 매트릭스 구조를 가지고 있다.)
          ic(type(xy)) # <class 'numpy.ndarray'>
          ic(xy.ndim) # xy.ndim: 2  # 차원
          ic(xy.shape) # xy.shape: (2922, 6) # 행렬의 갯수
          self.x_data = xy[:, 1:-1]# 해당 날짜 기후요소 4개=> 슬라이싱 문법
          self.y_data = xy[:, [-1]]# 해당 날짜 배추가격 => 인덱싱 문법
     
        
     #모델생성
     def create_model(self):
          # 텐서 모델 초기화(모델 템플릿생성)
          model =tf.global_variables_initializer()
          
          # 확률변수 데이터
          self.preprocessing()
          
          # 선형식제작 y= Wx+b
          #값 초기화 shape=>열
          X = tf.placeholder(tf.float32, shape=[None, 4])
          Y = tf.placeholder(tf.float32, shape=[None, 1])
          W = tf.Variable(tf.random_normal([4,1]), name ="weight")
          b = tf.Variable(tf.random_normal([1]), name="bias")
          hypothesis =tf.matmul(X, W)+ b #가설식 세우기,행열연산
          #=>matmul은 행렬곱 함수
          
          # 손실함수
          cost =  tf.reduce_mean(tf.square(hypothesis - Y))# 비용함수 설정 예측값 hypothesis에서 실제값 Y를 빼준다
          #=> reduce_mean 열단위로 평균을 내는 함수
          #=> square 제곱 함수
          
          # 최적화(Optimizer)-알고리즘
          #최적화 함수 설정 (학습률: 0.000005) 학습률은 데이터 정제의 정도이나, 유형에 따라 다르게 설정
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)#경사하강법(.train.GradientDescentOptimizer)
          train = optimizer.minimize(cost)#
          
          # 세션생성
          #세션값 만들어서 세션에 따라 학습수행
          sess = tf.Session()
          sess.run(tf.global_variables_initializer())
          
          # 트레이닝
          # 500번의 단계마다 진행 상황 확인하는 과정
          for step in range(100001):
               cost_, hypo_, _ = sess.run([cost,hypothesis,train], feed_dict={X: self.x_data, Y: self.y_data})
               if step % 500 == 0:
                    print("# %d 손실비용: %d" %(step, cost_))
                    print("-배추가격: %d" %(hypo_[0]))
          
          #모델 저장          
          saver = tf.train.Saver()
          saver.save(sess, os.path.join(self.model,'cabbage', 'model'),global_step=1000)
          print("학습된 모델을 저장했습니다.")
     
                
     
if __name__ == '__main__':
     tf.disable_v2_behavior()
     s=CabbageModel()
     s.create_model()