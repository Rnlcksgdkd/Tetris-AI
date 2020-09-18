import time
import copy
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import random
from Visualizer import Visualizer
from Tetris import Tetris
from collections import deque
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential

EPISODE = 1000000
GAME_VELOCTY = 0.000001
ACTION_VELCOCITY = 0.000001


UNIT = 30  # 픽셀 수
HEIGHT = 10  # 그리드 세로
WIDTH = 10   # 그리드 가로
MID = (WIDTH / 2 - 1) * UNIT # 블록 시작 점


#ret = [[0] * 84 for _ in range(84)]


## DQN 알고리즘을 이용한 테트리스 AI 학습 모듈
class DQN_Learner:
    def __init__(self, load_mode = False , filename = [], epsi = 1, ):

        ## 1. 테트리스 생성 ,  기본적으로 인공지능 안에 테트리스 환경이 정의되있음
        self.tetris = Tetris()
        self.numBlocktype = self.tetris.numType         # 게임에 이용할 블록 타입 숫자

        ## 2. 학습 모드 설정
        self.load_mode = False     # 학습을 이어서 하는 모드
        self.test_mode = False  # 학습된 파라미터 테스트 모드
        self.load_memory = False       # 학습에 이용할 메모리를 로드


        # 로드 기능 (입실론 값 설정)
        if self.load_mode:
            self.epsilon = 0.5
            for i in range(self.numBlocktype):
                self.models[i].load_weights("./save_model/DQN_v6.4_epi_127500_type_%d.h5" % i )

        # 테스트 기능
        if self.test_mode:
            self.tetris.DQN_TEST = True
            self.epsilon = 0.001     # 무 작위성 2프로
            for i in range(self.numBlocktype):
                self.models[i].load_weights("./save_model/DQN_v7.5_epi_902500_type_%d.h5" % i )

        ## 3. 신경망 정의 및 생성
        self.state_size = (10, 20, 1)               # 신경망의 Input , 상태변수 10*20 의 Bool 형태 변수
        self.action_space = [9, 17, 34, 34]         # 신경망의 output 리스트 , 각 블록마다 가능한 Action 의 수

        self.optimizers = []
        for i in range(self.numBlocktype):
            self.optimizers.append(self.optimizer(i))

        self.models = self.build_models()
        self.target_models = self.build_models()
        self.update_target_model()


        ## 4. 신경망 파라미터 설정

        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 0.5, 0.1
        self.exploration_steps = 3500000.
        #self.progress_rate = 0
        self.episodes = 1000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.train_start = 20000
        self.update_target_rate = 10000
        self.discount_factor = 0.95

        ## 학습을 위한 메모리 / 신경망 구성 : 각각 블록타입마다 신경망 / 타겟신경망 / 옵티마이져 / 메모리 존재

        # 리플레이 메모리
        self.batch_size = 32

        ## 5. 학습 결과 시각화를 위한 변수 정의
        self.SCORES = []            # 획득 점수
        self.AVG_Q = []             # 평균 Q 값
        self.AVG_LOSS = []          # 평균 Loss 값
        self.AVG_REWARD = []        # 평균 Reward 값
        self.STEPS = []             # 게임당 평균 스텝 수

        self.avg_q_max, self.avg_loss = 0, 0

       # 로드할 메모리가 있다면
        if self.load_memory:
            with open('Fmemory.txt', 'rb') as f:
                self.Fmemorys = pickle.load(f)


        self.memorys = []
        for i in range(self.numBlocktype):
            self.memorys.append(deque(maxlen=10000))
        self.no_op_steps = 30


        # 학습 횟수 카운터
        self.learning_cnt = [ 0 for __ in range(self.numBlocktype)]

        '''
        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")
        '''

    ## 신경망 관련 함수 ##

    # 1. Optimzier 함수 : 최적화 함수를 직접 정의.. 굳이 필요한가는 나도 잘 모름

    # 2. Build_model 함수  :  신경망 구성 함수 , 각 블록타입 마다 각 신경망을 생성 , 블록 타입에 따라 Action 수가 다름

    # 3.

    # Huber Loss 를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self, index):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.models[index].output

        a_one_hot = K.one_hot(a, self.action_space[index])
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        # lr = 0.00025
        optimizer = RMSprop(lr=0.0005, epsilon=0.01)
        updates = optimizer.get_updates(self.models[index].trainable_weights, [], loss)


        train = K.function([self.models[index].input, a, y], [loss], updates=updates)



        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self, action_size):

        model = Sequential()

        # Size 20 * 10 일 떄
        model.add(Conv2D(48, (3, 6), strides=(1, 2), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(32, (2, 2), strides=(2, 2), activation='relu'))

        # # Size 10 * 10 일 때
        # model.add(Conv2D(48, (2, 4), strides=(2, 2), activation='relu',
        #                  input_shape=self.state_size))
        # model.add(Conv2D(32, (4, 2), strides=(1, 1), activation='relu'))

        model.add(Flatten())
        model.add(Dense(60, activation='relu'))
        model.add(Dense(action_size))
        model.summary()
        return model



    # 학습 완료 후 결과 시각화 함수
    def plot_Train(self):

        vis  = Visualizer()
        vis.plot_SCORE(self.SCORES)
        vis.plot_Reward(self.AVG_REWARD)
        vis.plot_Qmax(self.AVG_Q)
        vis.plot_Loss(self.AVG_LOSS)
        vis.plot_Steps(self.STEPS)

        print("학습 완료!")

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):

        for i in range(self.numBlocktype):
            self.target_models[i].set_weights(self.models[i].get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history, type):


        history = np.float32(history)

        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space[type])
        else:
            q_value = self.models[type].predict(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, type):
        self.memorys[type].append((history, action, reward, next_history))

    # 전체 메모리 크기 계산 함수
    def getTotalMemory(self):

        ret = 0
        for i in range(self.numBlocktype):
            ret += len(self.memorys[i])

        return ret

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):


        # 학습함에 따라 탐험 비율 감소
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 학습할 블록 타입 랜덤 지정
        type = random.randrange(0, self.numBlocktype)
        mini_batch = random.sample(self.memorys[type], self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.target_models[type].predict(next_history)

        for i in range(self.batch_size):
            target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizers[type]([history, action, target])
        self.avg_loss += loss[0]


        self.learning_cnt[type] += 1

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        tf.train.AdamOptimizer
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    # # 실제 Q 값 을 확인하기 위한 UI 구현
    # def show_Qvalue(self, State):
    #
    #     QTable= self.model.predict(State)
    #
    #     Q_boards = [ [] for _ in range(len(QTable))]
    #
    #     for i in range(4):
    #         for j in range(10):
    #             # UI 표현
    #             Q_boards[i] = self.canvas.create_text( (WIDTH + 5 + j) * UNIT,  HEIGHT*UNIT/4*i ,
    #                                             fill="indianred1",
    #                                             font="Times 15 bold",
    #                                             text= str(int(QTable[i*10 + j])) )
    #     return Q_boards


    # 저장된 메모리를 이용하여 신경망 학습
    def learn_use_load_memory(self):

        # 학습할 블록 타입 랜덤 지정
        type = random.randrange(0, self.numBlocktype)
        mini_batch = random.sample(self.Fmemorys[type], self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward = [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0])
            next_history[i] = np.float32(mini_batch[i][3])
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])

        target_value = self.target_models[type].predict(next_history)

        for i in range(self.batch_size):
            target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizers[type]([history, action, target])
        self.avg_loss += loss[0]



    # 학습 결과를 저장하는 함수
    def save_LearnResult(self):

        # 학습 경과를 png 파일로 저장
        self.plot_Train()

        # 학습 결과를 csv 파일로 저장
        res = [1,2,3,4,5]

        # 학습 소요 시간
        learn_time = 0

        # 학습 전 설정된 파라미터들
        # BlockType 들?

        # 평균 Step Reward
        # 생존 Step

        # df = pd.DataFrame([ls], columns=['Name', 'Num', 'Layer_num', 'Nodesize', 'Dropout', 'Learningrate', 'Rmse'])
        # res_df = pd.concat([res_df, df], ignore_index=True)
        # res_df.to_csv('DQN_Result.csv')
        # 전체소요시간 ,
        pass


    # 신경망(혹은 무작위) 으로 얻은 행동을 진행
    def Play(self, action):

        next_map = self.tetris.line_step(action)
        reward = self.tetris.reward

        return next_map , reward


    # DQN 방식으로 학습
    def DQN_Learn(self, GLOBAL_STEPS):

        state = pre_processing(self.tetris.map)
        history = np.reshape([state], (1, 10, 20, 1))

        start_time = time.time()
        action_time = time.time()
        global_step = 0

        # 학습 시각화를 위한 변수세팅
        batch_SCORE = 0
        batch_Loss = 0
        batch_Qmax = 0
        batch_Reward = 0
        batch_Steps = 0

        # for i in range(400000):
        #     agent.avg_loss = 0
        #     agent.learn_use_load_memory()
        #
        #     if i % 1000 == 0:
        #         print("Learning_cnt : {} ,  Loss : {}".format(i , round(agent.avg_loss/agent.batch_size,3)) )
        #
        # for i in range(agent.numBlocktype):
        #     agent.models[i].save_weights("./save_model/DQN_v7.3_type_%d.h5" % (i))


        ## 학습 메인 루프 절차
        # 0. 학습 시간 = Episode or global_step 으로 제한
        # 1. 상태 -> 행동 얻고
        # 2. 행동 수행 -> 현상태, 다음상태, 행동, 보상 저장 -> 메모리로
        # 3. 학습 진행 -> 일정 주기 마다 타겟 신경망 업데이트

        epi = 0

        while True:
            epi += 1

            # 에피소드 마다 초기화
            step = 0
            avg_reward = 0
            score = 0

            # 학습 종료시 학습 결과 그래프로 출력하고 종료
            if self.epsilon < 0.1:
                self.save_LearnResult()
                break

            while True:
                # end_time = time.time()
                if 1:  # end_time - action_time >= ACTION_VELCOCITY:


                    # 1. 현재 상테에서 행동 및 보상을 받고 step 을 진행

                    score = self.tetris.score
                    batch_SCORE += score

                    # 블록 타입에 따라 신경망 -> 행동
                    type = self.tetris.curr_block[1]
                    action = self.get_action(history, type)
                    # agent.tetris.showQ(agent.models[type].predict(history))
                    # print("Type1: ", type )
                    # 테트리스에서 게임 진행 및 Reward 계산
                    global_step += 1
                    step += 1
                    self.tetris.line_step(action)
                    # print("라인스텝이후 타입: ", agent.tetris.curr_block[1])
                    reward = self.tetris.reward

                    avg_reward += reward
                    batch_Reward += reward

                    if self.test_mode:
                        print("Step : {} , Type : {} , Action : {}  , Reward : {} ".format(step, type, action, reward))

                    # 다음 상태 전처리
                    next_state = pre_processing(self.tetris.map)
                    next_history = np.reshape([next_state], (1, 10, 20, 1))

                    # 학습 진행을 보기 위해 Q 값 저장 : 이건 Random 행동을 했냐 안했냐랑 상관이 없음
                    q = np.amax(self.models[type].predict(np.float32(history))[0])
                    self.avg_q_max += q
                    batch_Qmax += q

                    batch_Steps += 1

                    # 초기부분만 학습하도록 설정
                    self.append_sample(history, action, reward, next_history, type)

                    ## 타입에 맞게 신경망으로 이어지는지, 해당 메모리를 잘 사용하는지 테스트용 코드
                    # print('type : {} , action : {} , reward : {} , type_len_memory : {}'.format(type, action, reward, len(agent.memorys[type])))

                    # if agent.tetris.Rsignal:
                    #
                    #     agent.tetris.reset()

                    # 신경망 학습
                    if not self.test_mode:
                        if self.getTotalMemory() > 15000:
                            self.train_model()

                    # 학습용 메모리가 있다면 일정 스텝마다 고정된 메모리를 학습하도록
                    if self.load_memory and global_step % 10 == 0:
                        self.learn_use_load_memory()

                    # if global_step >= agent.train_start:
                    #     agent.train_model()

                    # 타겟 신경망 업데이트
                    if global_step % agent.update_target_rate == 0:
                        self.update_target_model()

                    history = next_history
                    action_time = time.time()

                batch_Loss += self.avg_loss

                if 1:  # end_time - start_time >= GAME_VELOCTY:
                    # game over
                    if self.tetris.Rsignal:

                        # 1000 에피스모드마다 학습 결과 시각화를 위한 변수들 초기화
                        if epi % agent.episodes == 0:
                            batch_SCORE = round(batch_SCORE / agent.episodes, 3)
                            batch_Loss = round(batch_Loss / (batch_Steps * agent.batch_size), 3)
                            batch_Qmax = round(batch_Qmax / batch_Steps, 3)
                            batch_Reward = round(batch_Reward / batch_Steps * 10, 3)
                            batch_Steps = round(batch_Steps / agent.episodes, 3)

                            # 중간 결과 저장
                            agent.SCORES.append(batch_SCORE)
                            agent.AVG_LOSS.append(batch_Loss)
                            agent.AVG_Q.append(batch_Qmax)
                            agent.AVG_REWARD.append(batch_Reward)
                            agent.STEPS.append(batch_Steps)
                            # 확인용
                            print("중간 학습 결과 : Score {} , Avg_Loss {} , Avg_Qmax {} , Avg_Reward {} ".format(batch_SCORE,
                                                                                                            batch_Loss,
                                                                                                            batch_Qmax,
                                                                                                            batch_Reward))

                            # 학습 시각화를 위한 변수 초기화
                            batch_SCORE = 0
                            batch_Loss = 0
                            batch_Qmax = 0
                            batch_Reward = 0
                            batch_Steps = 0

                        '''
                        if global_step > agent.train_start:
                            stats = [tetris.score, agent.avg_q_max / float(global_step), global_step,
                                     agent.avg_loss / float(global_step)]
                            for i in range(len(stats)):
                                agent.sess.run(agent.update_ops[i], feed_dict={
                                    agent.summary_placeholders[i]: float(stats[i])
                                })
                            summary_str = agent.sess.run(agent.summary_op)
                            agent.summary_writer.add_summary(summary_str, epi + 1)
                        '''
                        print(
                            'episode:{}, score:{}, epsilon:{}, global step:{}, loss : {},  avg_qmax:{}, memory:{}, avg_reward:{}, game_Step:{}'.
                            format(epi, score, agent.epsilon, global_step, agent.avg_loss / float(agent.batch_size),
                                   agent.avg_q_max / float(step), agent.getTotalMemory(), float(avg_reward / step),
                                   step))

                        agent.avg_q_max, agent.avg_loss = 0, 0

                        break
                    else:
                        buffer = agent.tetris.step(0)

                    start_time = time.time()

            # epi 수가 10000이 넘어가면 1000 단위 마다
            if epi > 10000 and epi % 2500 == 0:
                for i in range(agent.numBlocktype):
                    agent.models[i].save_weights("./save_model/DQN_v7.5_epi_%d_type_%d.h5" % (epi, i))

                for i in range(agent.numBlocktype):
                    print("%d 번째 블록 Learning_cnt : %d" % (i, agent.learning_cnt[i]))

        pass


def pre_processing(curr_map):
    copy_map = copy.deepcopy(curr_map)
    # ny, nx = 4.20, 10.5
    # for n in range(20):
    #     for m in range(8):
    #         for i in range(int(n * ny), int(n * ny + ny)):
    #             for j in range(int(m * nx), int(m * nx + nx)):
    #                 ret[i][j] = copy_map[n][m]
    return copy_map


if __name__ == "__main__":

    agent = DQNAgent()



    state = pre_processing(agent.tetris.map)
    history = np.reshape([state], (1, 10, 10, 1))

    start_time = time.time()
    action_time = time.time()
    global_step = 0

    # 학습 시각화를 위한 변수세팅
    batch_SCORE = 0
    batch_Loss = 0
    batch_Qmax = 0
    batch_Reward = 0
    batch_Steps = 0

    # for i in range(400000):
    #     agent.avg_loss = 0
    #     agent.learn_use_load_memory()
    #
    #     if i % 1000 == 0:
    #         print("Learning_cnt : {} ,  Loss : {}".format(i , round(agent.avg_loss/agent.batch_size,3)) )
    #
    # for i in range(agent.numBlocktype):
    #     agent.models[i].save_weights("./save_model/DQN_v7.3_type_%d.h5" % (i))

    ## 학습 메인 루프 절차
    # 0. 학습 시간 = Episode or global_step 으로 제한
    # 1. 상태 -> 행동 얻고
    # 2. 행동 수행 -> 현상태, 다음상태, 행동, 보상 저장 -> 메모리로
    # 3. 학습 진행 -> 일정 주기 마다 타겟 신경망 업데이트

    for epi in range(EPISODE):

        # 에피소드 마다 초기화
        step = 0
        avg_reward = 0
        score = 0



        # 학습 종료시 학습 결과 그래프로 출력하고 종료
        if not agent.test_mode and agent.epsilon < 0.1:
            agent.plot_Train()
            break

        while True:
            #end_time = time.time()
            if 1: #end_time - action_time >= ACTION_VELCOCITY:
                # 1. 현재 상테에서 행동 및 보상을 받고 step 을 진행

                score = agent.tetris.score
                batch_SCORE += score

                # 블록 타입에 따라 신경망 -> 행동
                type = agent.tetris.curr_block[1]
                action = agent.get_action(history, type)
                #agent.tetris.showQ(agent.models[type].predict(history))
                #print("Type1: ", type )
                # 테트리스에서 게임 진행 및 Reward 계산
                global_step += 1
                step += 1
                agent.tetris.line_step(action)
                #print("라인스텝이후 타입: ", agent.tetris.curr_block[1])
                reward = agent.tetris.reward

                avg_reward += reward
                batch_Reward += reward



                if agent.test_mode:
                    print("Step : {} , Type : {} , Action : {}  , Reward : {} ".format(step, type, action, reward))


                # 다음 상태 전처리
                next_state = pre_processing(agent.tetris.map)
                next_history = np.reshape([next_state], (1, 10, 10, 1))

                # 학습 진행을 보기 위해 Q 값 저장 : 이건 Random 행동을 했냐 안했냐랑 상관이 없음
                q = np.amax(agent.models[type].predict(np.float32(history))[0])
                agent.avg_q_max += q
                batch_Qmax += q

                batch_Steps += 1

                # 초기부분만 학습하도록 설정

                agent.append_sample(history, action, reward, next_history, type)

                ## 타입에 맞게 신경망으로 이어지는지, 해당 메모리를 잘 사용하는지 테스트용 코드
                #print('type : {} , action : {} , reward : {} , type_len_memory : {}'.format(type, action, reward, len(agent.memorys[type])))

                # if agent.tetris.Rsignal:
                #
                #     agent.tetris.reset()

                # 신경망 학습
                if not agent.test_mode:
                    if agent.getTotalMemory() > 15000:
                        agent.train_model()

                # 학습용 메모리가 있다면 일정 스텝마다 고정된 메모리를 학습하도록
                if agent.load_memory and global_step % 10 == 0:
                    agent.learn_use_load_memory()

                # if global_step >= agent.train_start:
                #     agent.train_model()

                # 타겟 신경망 업데이트
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

                history = next_history
                action_time = time.time()

            batch_Loss += agent.avg_loss

            if 1: #end_time - start_time >= GAME_VELOCTY:
                # game over
                if agent.tetris.Rsignal:

                    # 1000 에피스모드마다 학습 결과 시각화를 위한 변수들 초기화
                    if epi % agent.episodes == 0:

                        batch_SCORE = round(batch_SCORE / agent.episodes, 3)
                        batch_Loss = round(batch_Loss / (batch_Steps*agent.batch_size), 3)
                        batch_Qmax = round(batch_Qmax / batch_Steps, 3)
                        batch_Reward = round(batch_Reward / batch_Steps * 10, 3)
                        batch_Steps = round(batch_Steps / agent.episodes, 3)

                        # 중간 결과 저장
                        agent.SCORES.append(batch_SCORE)
                        agent.AVG_LOSS.append(batch_Loss)
                        agent.AVG_Q.append(batch_Qmax)
                        agent.AVG_REWARD.append(batch_Reward)
                        agent.STEPS.append(batch_Steps)
                        # 확인용
                        print("중간 학습 결과 : Score {} , Avg_Loss {} , Avg_Qmax {} , Avg_Reward {} ".format(batch_SCORE,
                                                                                                        batch_Loss,
                                                                                                        batch_Qmax,
                                                                                                        batch_Reward))

                        # 학습 시각화를 위한 변수 초기화
                        batch_SCORE = 0
                        batch_Loss = 0
                        batch_Qmax = 0
                        batch_Reward = 0
                        batch_Steps = 0


                    '''
                    if global_step > agent.train_start:
                        stats = [tetris.score, agent.avg_q_max / float(global_step), global_step,
                                 agent.avg_loss / float(global_step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, epi + 1)
                    '''
                    print('episode:{}, score:{}, epsilon:{}, global step:{}, loss : {},  avg_qmax:{}, memory:{}, avg_reward:{}, game_Step:{}'.
                          format(epi, score, agent.epsilon, global_step, agent.avg_loss / float(agent.batch_size),
                                 agent.avg_q_max / float(step), agent.getTotalMemory(), float(avg_reward/step), step))

                    agent.avg_q_max, agent.avg_loss = 0, 0

                    break
                else:
                    buffer = agent.tetris.step(0)

                start_time = time.time()

        # epi 수가 10000이 넘어가면 1000 단위 마다
        if epi > 10000 and epi % 2500 == 0:
            for i in range(agent.numBlocktype):
                agent.models[i].save_weights("./save_model/DQN_v7.5_epi_%d_type_%d.h5" % (epi , i) )

            for i in range(agent.numBlocktype):
                print("%d 번째 블록 Learning_cnt : %d" %(i, agent.learning_cnt[i]) )







