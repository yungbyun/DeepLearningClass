import tensorflow as tf
import numpy as np
from abc import abstractmethod


'''

'''
class EnsembleCore:
    sess = None
    networks = None

    @abstractmethod
    def set_networks(self, sess, num_of_network):
        pass

    @abstractmethod
    def get_number_of_segment(self, seg_size):
        pass

    @abstractmethod
    def get_next_segment(self, seg_size):
        pass

    @abstractmethod
    def load_db(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    def create_networks(self, sess, class_name, net_name, num_of_network):
        net = []
        for num in range(num_of_network):
            net_name = net_name + str(num)
            net.append(class_name(sess, net_name))   # MyNetwork here
        self.networks = net

    # main 함수
    def learn_ensemble(self, num_of_network, epoch_num, segment_size):
        tf.set_random_seed(777)  # reproducibility

        self.load_db()  # virtual

        self.sess = tf.Session()

        self.set_networks(self.sess, num_of_network)  # virtual

        self.sess.run(tf.global_variables_initializer())

        # train my model
        avg_err_list = np.zeros(num_of_network)  # 아래 for 루프에 있었는데 여기로 옮김 (여기가 맞는 듯 하다.)
        print('\nStart learning ensemble:')
        for epoch in range(epoch_num):
            # 세그먼트 크기를 줄테니 몇 개의 세그먼트로 나눠지는지 알려다오.
            num_of_segment = self.get_number_of_segment(segment_size)  # virtual

            # 처음 데이터를 100개를 읽어 최적화함.
            # 그 다음 100개 데이터에 대하여 수행.
            # 이를 모두 550번 수행하면 전체 데이터 55,000개에 대해 1번 수행하게 됨.
            # 아래 for 문장이 7개의 모델 각각에 대해 전체 데이터로 1번 실행(학습)함.
            for i in range(num_of_segment):
                x_segment, y_segment = self.get_next_segment(segment_size)  # virtual

                # 특정 세그먼트 하나를 이용하여 7개 객체들을 각각 한번 학습(미분, W와 b 조정) 시킴. 에러 저장
                for m_idx, model in enumerate(self.networks):
                    # print('Training object:', m_idx)
                    # 아래 에러는 일부분(100개)에 대한 것이므로 전체에 대한 에러를 구하려면 550으로 나누어주어야 함.
                    cost, _ = model.learn_with_a_segment(x_segment, y_segment, 0.7)  # 한번 미분
                    avg_err_list[m_idx] += cost / num_of_segment

                print('Epoch', epoch, ', 세그먼트', i, ' 데이터로 학습(미분, W와 b를 조정)한 후 7개 모델 각각 평균 오류:\n ', avg_err_list)

        print('Done!')

    # 10,000개의 테스트 데이터로 인식률을 구함.
    def evaluate_all_models(self):
        #test_size = len(self.db.test.labels)

        #predictions = np.zeros(test_size * 10).reshape(test_size, 10)
        # print(predictions)

        x_data, y_data = self.get_test_data()

        # 학습이 완료된 7개의 모델 각각에게 실행하도록 함.
        for index, model in enumerate(self.networks):
            acc = model.evaluate(x_data, y_data, 1.0)
            #print('images', self.db.test.images, 'labels', self.db.test.labels)
            print('Network#:', index, 'Accuracy:', acc)

            # p에는 라벨 스트링이 10,000개 들어있음. 7개의 모델에 대하여 수행하니 루프가 끝나면 7만개가 들어있음.
            #p = model.test(self.db.test.images) # 이미지 10,000개
            #predictions += p

        #print(len(predictions))

        # 아래 코드 확인 필요!! predictions에 p(10,000개 라벨)가 7번 더해지므로 7만개의 라벨이 들어가는데 아래 오른쪽 코드에서 비교하고 있는
        # 라벨은 만개.. 정답 라벨을 7번 반복한 7만개여야 하는데.. predictions도 만개 공간만 갖도록 위에서 정의되어 있는 것도 문제.
        # 설사 이게 맞더라도 왜 이것이 Ensemble accuracy가 되는지 이해가 안됨.
        #hit_record = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.db.test.labels, 1))
        #ensemble_accuracy = tf.reduce_mean(tf.cast(hit_record, tf.float32))
        #print('Ensemble accuracy:', self.sess.run(ensemble_accuracy))

