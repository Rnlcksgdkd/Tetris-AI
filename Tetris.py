
import time
import copy
import pickle
import numpy as np
import random
from collections import deque
import tkinter as tk
from PIL import ImageTk , Image


PhotoImage = ImageTk.PhotoImage # 이미지
np.random.seed(1) # 랜덤시드
UNIT = 30  # 픽셀 수
HEIGHT = 20  # 그리드 세로
WIDTH = 10   # 그리드 가로
MID = (WIDTH / 2 - 1) * UNIT # 블록 시작 점

# 명령키
DOWN = 0
LEFT = 1
RIGHT = 2
ROTATE = 3
DROP = 4

# DOWN, LEFT, RIGHT 에 해당하는 변화량
dy = [UNIT, 0, 0]       # action에 해당하는 y좌표 plus값
dx = [0, -UNIT, +UNIT]   # action에 해당하는 x좌표 plus값

## 블록 구조 : 7가지

# 1.O
block1_dx = [[0, 1, 1, 0]] * 4
block1_dy = [[0, 0, 1, 1]] * 4

# 2.I
block2_dx = [[0, 0, 0, 0], [0, 1, 2, 3]] * 2
block2_dy = [[0, 1, 2, 3], [1, 1, 1, 1]] * 2

# 3.S
block3_dx = [[0, 0, 1, 1], [0, 1, 1, 2]] * 2
block3_dy = [[0, 1, 1, 2], [2, 2, 1, 1]] * 2

# 4.Z
block4_dx = [[1, 1, 0, 0], [0, 1, 1, 2]] * 2
block4_dy = [[0, 1, 1, 2], [1, 1, 2, 2]] * 2

# 5.T
block5_dx = [[0, 0, 0, 1], [0, 1, 2, 1], [0, 1, 1, 1], [0, 1, 2, 1]]
block5_dy = [[0, 1, 2, 1], [1, 1, 1, 2], [1, 0, 1, 2], [1, 1, 1, 0]]

# 6.L
block6_dx = [[0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 1, 1], [0, 1, 2, 2]]
block6_dy = [[0, 1, 2, 2], [2, 1, 1, 1], [0, 0, 1, 2], [1, 1, 1, 0]]

# 7.J
block7_dx = [[0, 1, 1, 1], [0, 0, 1, 2], [0, 0, 0, 1], [0, 1, 2, 2]]
block7_dy = [[2, 2, 1, 0], [1, 2, 2, 2], [2, 1, 0, 0], [1, 1, 1, 2]]

blocks_dx = [ block1_dx, block2_dx, block6_dx, block4_dx, block7_dx, block3_dx, block5_dx ]
blocks_dy = [ block1_dy, block2_dy, block6_dy, block4_dy, block7_dy, block3_dy, block5_dy ]

#blocks_dx = [ block1_dx, block2_dx, block6_dx, block7_dx ]
#blocks_dy = [ block1_dy, block2_dy, block6_dy, block7_dy  ]

# 각 원소에 UNIT 을 곱하기 위해 np 로 변환 후 다시 역변환
UNITblocks_dx = (UNIT*(np.array(blocks_dx))).tolist()
UNITblocks_dy = (UNIT*(np.array(blocks_dy))).tolist()

PLUS_SCORE = 10.0

basic_counter_str = 'test : '
basic_score_str = 'score : '

# 파라미터
A = 1.6         # 블록갯수
B = 1           # 라인 클리어 (사용 x)
C = -4.3          # 높이 편차
D = -3.2        # 세로 3칸 이상 빈칸
E = -3.4        # 막힌 블록
F = -0.5        # 단면적
G = -1          # 최대 높이


## 테트리스 : 수동 키 조작 ##

class Tetris(tk.Tk) :

    def __init__(self):
        super(Tetris, self).__init__()

        # 게임 신호 관련
        self.clk = 0 
        self.Np = [MID, 0]  # 현재 블록 위치
        
        self.pause = False      # 중지 시그널
        self.nowcarrying = False  # 블록 이동 시그널
        
        # 게임 외적으로 필요한 시그널 : 게임 자체에는 영향이 없음
        self.Ssignal = False  # 저장 시그널 : 수동 행동 -> 메모리 형태로 저장하기 위한 시그널
        self.Rsignal = False  # 게임 리셋 시그널 : 게임이 리셋 됬다는 것을 알리는 시그널임 , 게임 자체에는 영향이 없다


        # DQN 관련
        self.DQN_TEST = False   # DQN_AGENT 학습 테스트 시그널 , 켜지면 rendering 으로 게임 진행 상황을 볼 수 있도록 설정

        # 점수 관련
        self.score = 0.0
        self.counter = 0

        # Reward 계산 요소
        self.reward = 0
        self.poten = 0.0
        self.pv_poten = 0.0
        self.combo = 0
        self.comboline = 0

        # DQN 관련 인공신경망 테스트용
        #self.Qboard = []

        # 행동 정의
        self.action_space = ['d', 'l', 'r']
        self.action_size = len(self.action_space)

        self.action_Line = []
        self.Agent_action_space = [1, 2, 3, 4, 5, 6, 7, 8] # 각 라인
        self.Agent_action_size = len(self.Agent_action_space)

        # 블록 관련
        self.Type = ["Rectang", "Linear"]
        self.color = [ "red", "blue", "green", "yellow", "purple", 'orange', 'cyan']
        self.block_colors = len(self.color)
        self.block_types = len(self.Type)


        self.numType = 7            # 블록 타입 숫자 (원래는 7)

        self.block = []
        self.block_pos = []

        # 블록 꾸러미 리스트
        self.order_blocks = []

        # 현재 이동중인 블록을 가르키는 변수들
        self.curr_block = [[],[],[]]
        self.curr_block[0] = -1  # 현재블록 타입 ( 색깔 )
        self.curr_block[1] = -1
        self.curr_block[2] = 0

        # 메모리 설정
        if self.Ssignal:
            self.memorysize = 1000
            self.memorys = []
            for i in range(self.numType):
                self.memorys.append(deque(maxlen=250))
            print("수동 메모리 작성 시작합니다")


        # 맵 정보
        self.canvas, self.counter_board, self.score_board = self._build_canvas()


        self.map = [[0] * WIDTH for _ in range(HEIGHT)]
        self.canvas.pack()
        self.canvas.focus_set()

        # 수동 키입력 부분 #
        self.canvas.bind("<Down>", lambda _: self.move(DOWN))
        self.canvas.bind("<Left>", lambda _: self.move(LEFT))
        self.canvas.bind("<Right>", lambda _: self.move(RIGHT))

        self.canvas.bind("<Up>", lambda _: self.Rotate())
        self.canvas.bind("<Return>", lambda _: self.hard_drop())
        self.canvas.bind("<Delete>", lambda _: self.Pause())

        # 게임 루프 #
        self.game_loop()       # 수동 키





    ## 캔버스 생성단 - tk 라이브러리 이용
    def _build_canvas(self):

        # 공간 분할 #
        canvas = tk.Canvas(self, bg='black',
                           height=HEIGHT * UNIT,
                           width= (WIDTH + 29) * UNIT)
        counter_board = canvas.create_text((WIDTH + 3) * UNIT, int(HEIGHT / 4 * UNIT),
                                           fill="white",
                                           font="Times 10 bold",
                                           text=basic_counter_str + str(int(self.counter)))

        score_board = canvas.create_text((WIDTH + 3) * UNIT, int(HEIGHT / 2 * UNIT),
                                         fill="white",
                                         font="Times 10 bold",
                                         text=basic_score_str + str(int(self.score)))


        # 그리드 생성
        for c in range(0, (WIDTH + 1) * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill='white')
        '''
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)
        '''
        return canvas, counter_board, score_board


    ### 블록 생성단  ###

    # 블록 꾸러미 발생 -> 블록 순서를 결정해준다 : 즉 고정된 순서가 아닌 블록 생성
    def assign_order_blocks(self):

        self.order_blocks = []

        # 랜덤 순서를 위한 마킹
        mark = [0 for __ in range(self.numType)]
        for i in range(self.numType):
            mark[i] = i

        # 블록 생성 순서 무작위 결정
        while len(self.order_blocks) < self.numType:
            idx = random.randrange(0, len(mark))
            self.order_blocks.append(mark[idx])
            del mark[idx]

        print(self.order_blocks)

    # 블록 발생 함수 : 블록 이동이 끝났으면 무작위 블록 생성
    def get_cur_block(self):

        # 리셋 했다는 신호가 오면 다시 순서를 배정
        if self.Rsignal:
            self.assign_order_blocks()
            self.Rsignal = False

        # 블록 꾸러미를 다 쓴 경우에는 다시 순서를 배정
        if len(self.order_blocks) == 0:
            self.assign_order_blocks()

        # 블록 꾸러미에 의해 타입을 결정 , 순서에서 삭제
        self.curr_block[1] = self.order_blocks[0]
        self.curr_block[0] = self.curr_block[1]
        self.curr_block[2] = 0  # 회전 타입 ( 디폴트 0 )
        del self.order_blocks[0]

        # 블록 시작 위치
        self.block_pos = [MID, 0]
        self.block = self.creat_Block(self.block_pos)


        return self.block

     # 기준점에 대하여 블록 생성 함수 : 현재 블록 색, 타입, 회전에 맞춰 블록을 캔버스에 그려주고 저장함
    def creat_Block(self, pos):

        # 현재 회전
        c_idx = self.curr_block[0]  # 색
        b_idx = self.curr_block[1]  # 블록타입
        r_idx = self.curr_block[2]  # 회전타입

        # print(b_idx, r_idx, blocks_dx[b_idx][r_idx][0])
        rect1 = self.canvas.create_rectangle(pos[0] + UNITblocks_dx[b_idx][r_idx][0],
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][0],
                                             pos[0] + UNITblocks_dx[b_idx][r_idx][0] + UNIT,
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][0] + UNIT,
                                             fill=self.color[c_idx], tag="rect")
        rect2 = self.canvas.create_rectangle(pos[0] + UNITblocks_dx[b_idx][r_idx][1],
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][1],
                                             pos[0] + UNITblocks_dx[b_idx][r_idx][1] + UNIT,
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][1] + UNIT,
                                             fill=self.color[c_idx], tag="rect")
        rect3 = self.canvas.create_rectangle(pos[0] + UNITblocks_dx[b_idx][r_idx][2],
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][2],
                                             pos[0] + UNITblocks_dx[b_idx][r_idx][2] + UNIT,
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][2] + UNIT,
                                             fill=self.color[c_idx], tag="rect")
        rect4 = self.canvas.create_rectangle(pos[0] + UNITblocks_dx[b_idx][r_idx][3],
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][3],
                                             pos[0] + UNITblocks_dx[b_idx][r_idx][3] + UNIT,
                                             pos[1] + UNITblocks_dy[b_idx][r_idx][3] + UNIT,
                                             fill=self.color[c_idx], tag="rect")
        block = [rect1, rect2, rect3, rect4]
        return block

    # 현재 블록 위치 리턴 (이거 굳이 필요는 없을 거같은데??)
    def _get_curr_block_pos(self):
        ret = []
        if self.nowcarrying == False:
            return ret

        for n in range(4):
            s = (self.canvas.coords(self.block[n]))
            y = int(s[1] / UNIT)
            x = int(s[0] / UNIT)
            ret.append([y, x])
        return ret


    ### 블록 이동부 : ACTION 관련 : move , Rotate, Drop , Pause    ###

    # 블록 이동 함수 : LEFT , DOWN , RIGHT 행동을 받아 수행 , possible_Action 함수를 통해 블록을 굳힘 / 라인클리어 / 게임 종료 체크 함수 이어짐
    def move(self, action):

            if self.nowcarrying:
                flag = self.possible_move(action)
            else:
                return 0

            if flag == 2:  # 블록 배치완료

                # 블록 배치 전 상태 기억
                if self.Ssignal:

                    total_memory_size = 0
                    for i in range(self.numType):
                        total_memory_size += len(self.memorys[i])

                    if total_memory_size >= self.memorysize:
                        print("메모리 작성 완료")

                        with open('memory.txt', 'wb') as f:
                            pickle.dump(self.memorys, f)

                        self.Ssignal = False

                    type = self.curr_block[1]
                    state = pre_processing(self.map)
                    history = np.reshape([state], (1, 10, 20, 1))

                    # 블록이 굳을 시 회전 타입과 x 좌표를 계산
                    rot = self.curr_block[2]
                    nx = (self.block_pos[0]/UNIT)

                    if self.curr_block[1] == 0:
                        action =  nx
                    elif self.curr_block[1] == 1:
                        action = rot*10 + nx
                    elif self.curr_block[1] == 2 or self.curr_block[1] == 3:
                        if rot == 0:
                            action = nx
                        elif rot == 1:
                            action = 9 + nx
                        elif rot == 2:
                            action = 17 + nx
                        elif rot == 3:
                            action = 26 + nx


                self.add_Canvas()

                # 라인체크 -> 라인지울수있으면 지운다
                if self.check_line().__len__() != 0:
                    cl = self.check_line()
                    self.comboline = self.clear_line(cl)  # 라인지우기 , REWARD 발생부분
                    time.sleep(0.1)
                    self.map = self.down_all_canvas(cl)  # 라인지웟으면 전체블록 이동
                else:
                    self.comboline = 0



                # 게임종료체크
                if self.is_game_end():
                    self.reset()
                    self.reward = -20

                # Reward 계산
                self.calPoten()
                self.calCombo()

                self.reward = self.poten - self.pv_poten + self.combo
                #print(self.reward)

                # 블록 배치 후 상태 기억
                if self.Ssignal:
                    next_state = pre_processing(self.map)
                    next_history = np.reshape([next_state], (1, 10, 20, 1))
                    self.memorys[type].append((history, action, self.reward, next_history))

                    print("type : {}, action : {} , reward : {} ".format(type ,action , self.reward))
                    print(type, " 번 째 블록 메모리 : ", len(self.memorys[i]))

            elif flag == 3:  # 이동 가능하다면
                if action == 0:  # 밑
                    for n in range(4):
                        self.canvas.move(self.block[n], 0, 30)
                    self.block_pos[1] += UNIT
                elif action == 1:  # 왼
                    for n in range(4):
                        self.canvas.move(self.block[n], -30, 0)
                    self.block_pos[0] -= UNIT

                elif action == 2:  # 오른
                    for n in range(4):
                        self.canvas.move(self.block[n], +30, 0)
                    self.block_pos[0] += UNIT

    # 일시정지 함수 : DELETE 키  -> STEP 함수를 막아주어 일시정지 OR 수동 조작 가능
    def Pause(self):

        if self.pause :
            self.pause = False
        else:
            self.pause = True

    # 회전 함수
    def Rotate(self):

        # 회전한 블록을 새로 생성하기 위해 위치를 받음
        pos = self.block_pos

        # 회전
        if self.curr_block[2] == 3:
            self.curr_block[2] = 0
        else:
            self.curr_block[2] += 1

        # 회전 가능 체크하기 위해 회전 좌표 계산
        b_idx = self.curr_block[1]  # 블록타입
        r_idx = self.curr_block[2]  # 회전타입
        pos1 = [pos[0] + UNITblocks_dx[b_idx][r_idx][0], pos[1] + UNITblocks_dy[b_idx][r_idx][0]]
        pos2 = [pos[0] + UNITblocks_dx[b_idx][r_idx][1], pos[1] + UNITblocks_dy[b_idx][r_idx][1]]
        pos3 = [pos[0] + UNITblocks_dx[b_idx][r_idx][2], pos[1] + UNITblocks_dy[b_idx][r_idx][2]]
        pos4 = [pos[0] + UNITblocks_dx[b_idx][r_idx][3], pos[1] + UNITblocks_dy[b_idx][r_idx][3]]
        posN = [pos1, pos2, pos3, pos4]

        # 회전가능 체크
        for i in range(4):
            x = int(posN[i][0] / UNIT)
            y = int(posN[i][1] / UNIT)

            # 회전 불가시 복귀
            if x >= WIDTH or x < 0 or y >= HEIGHT or self.map[y][x] == 1:
                if self.curr_block[2] == 0:
                    self.curr_block[2] = 3
                else:
                    self.curr_block[2] -= 1
                return 0

        # 회전 가능시 기존 블록 삭제 후 회전, 생성
        for n in range(4):
            self.canvas.delete(self.block[n])
        self.block = self.creat_Block(self.block_pos)



    # Drop 함수
    def hard_drop(self):

        if self.nowcarrying == False:
            return 0

        while self.possible_move(0) != 2:
            self.move(0)

        self.move(0)

    # 이동 여부 체크 ( RIGHT, LEFT, DOWN 에 대해)
    def possible_move(self, action):

        for n in range(len(self.block)):
            s = self.canvas.coords(self.block[n])
            y = s[1] + dy[action]
            x = s[0] + dx[action]

            # 범위밖 - stay
            if x >= WIDTH * UNIT or x < 0:
                return 1
            ny = int(y / UNIT)
            nx = int(x / UNIT)

            # 맨 밑에줄인경우
            if y >= HEIGHT * UNIT:
                return 2
            # 이동하려고한위치가 블록일경우
            if self.map[ny][nx] == 1:
                if action == 0:
                    return 2  # 다운일경우 -> ADD
                else:
                    return 1  # 왼쪽/오른쪽  -> 못감
        # 이동가능
        return 3

    # 일시정지 함수 : DELETE 키  -> STEP 함수를 막아주어 일시정지 OR 수동 조작 가능
    def Pause(self):

        if self.pause :
            self.pause = False
        else:
            self.pause = True


    ### 블록종료부 : 블록 굳히기 / 라인 클리어 부분  ###

    # 블록 굳히기
    def add_Canvas(self):


        # 블록 배치 완료시 , nowcarry 신호 변경 -> 블록 다시 생성
        self.nowcarrying = False

        for n in range(4):
            pos = self.canvas.coords(self.block[n])
            nx = int(pos[0] / UNIT)
            ny = int(pos[1] / UNIT)
            self.map[ny][nx] = 1

        return 0



    # 줄 클리어 체크
    def check_line(self):
        cl = []
        for y in range(HEIGHT):
            flag = True
            for x in range(WIDTH):
                if self.map[y][x] == 0:
                    flag = False
                    break
            if flag:
                cl.append(y)
        return cl

    # 라인 클리어
    def clear_line(self, cl):

        score= 0
        for crect in self.canvas.find_withtag("rect"):
            ny = int(self.canvas.coords(crect)[1] / UNIT)
            if cl.count(ny) != 0:
                nx = int(self.canvas.coords(crect)[0] / UNIT)
                self.map[ny][nx] = 0
                self.canvas.delete(crect)

        for y in cl:
            score = score + y - 1

        self.score += score



        return len(cl)

    # 클리어 후 블록 다운
    def down_all_canvas(self, cl):
        new_map = [[0] * WIDTH for _ in range(HEIGHT)]

        for crect in self.canvas.find_withtag("rect"):

            ny = int(self.canvas.coords(crect)[1] / UNIT)
            nx = int(self.canvas.coords(crect)[0] / UNIT)

            count = 0  # 얼마나 이동해야하는지
            for y in range(cl.__len__()):
                if ny < cl[y]:
                    count = count + 1

            self.canvas.move(crect, 0, UNIT * count)
            self.map[ny][nx] = 0
            new_map[ny + count][nx] = 1

        return new_map


    ### 게임종료 및 리셋 부분    ###

    # 게임 종료 체크 #
    def is_game_end(self):

        for n in range(2):
            for m in range(WIDTH):
                if self.map[n][m] == 1:
                    return True
        return False

    # 리셋 함수
    def reset(self):

        # for i in range(len(self.Qboard)):
        #     self.canvas.delete(self.Qboard[i])

        self.curr_block[1] = -1
        self.score = 0.0

        # Reward 계산 요소
        self.reward = 0
        self.poten = 0.0
        self.pv_poten = 0.0
        self.combo = 0
        self.comboline = 0

        # DQN 관련 인공신경망 테스트용
        #self.Qboard = []

        self.counter += 1
        self.canvas.itemconfigure(self.counter_board,
                                  text=basic_counter_str + str(int(self.counter)))
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))
        self.update()
        self.canvas.delete("rect")

        self._clear_map()

        self.Rsignal = True
        self.nowcarrying = False

        # self._add_Canvas()

    # map 지우기
    def _clear_map(self):
        for n in range(HEIGHT):
            for m in range(WIDTH):
                self.map[n][m] = 0




    ### 게임 진행 루프 부분 ###

    # 업데이트 및 속도 조절 함수
    def render(self):
        self.update()
        #time.sleep(0.3)    #학습시 주석처리 , 실제 게임 진행 보고싶으면 0.1 ~ 0.3 설정

    # 업데이트 및 속도 조절 함수
    def render_TEST(self):
        self.update()
        time.sleep(1)    #학습시 주석처리 , 실제 게임 진행 보고싶으면 0.1 ~ 0.3 설정

    # 메인 업데이트 함수
    def game_update(self):

        # rendering
        if self.DQN_TEST:
            self.render_TEST()
        else:
            self.render()

        # canvas 업데이트
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))


    # 수동 조작용 루프
    def game_loop(self):

        self.clk = self.clk + 1                     # 게임 내 클락

        if self.nowcarrying == False:
            self.block = self.get_cur_block()
            self.nowcarrying = True

        else:
            # 3 clk 에 한번 자동 이동
            if self.clk % 3 == 0:
                self.move(0)

        if self.clk > 10000:
            self.clk = 0

        self.after(300, self.game_loop)         # 재귀 루프


    ##========================##
    ###    DQN AI 추가 부분   ###
    ##========================##

    # Reward 를 계산하기 위한 점수 부분
    def calPoten(self):

        self.pv_poten = self.poten

        # 어드밴티지 요소 초기화
        a = 0  # a 는 현재 쌓인 블록의 숫자
        b = 0  # b 는 라인클리어

        # 패널티 요소 초기화
        c = 0  # c 는 평균 높이에 따른 편차
        d = 0  # d 는 3칸이상 세로 구덩이 숫자
        e = 0  # e 는 세로줄 기준으로 빈칸 숫자
        f = 0  # f 는 쌓인 블록에 대한 단면적
        g = 0  # g 는 최대 높이

        # C 구하기 위한 임시 변수
        lineheights = [0 for _ in range(WIDTH)]

        for x in range(WIDTH):
            lineflag = True  # 라인별 최대 높이를 구하기 위한 flag
            deepflag = True

            for y in range(HEIGHT):

                # A. 전체 블록의 갯수 구하기
                if self.map[y][x] == 1:
                    a += 1

                    # C 계산을 위해 세로 라인별 높이 저장
                    if lineflag:
                        lineheights[x] = (-y + HEIGHT)
                        lineflag = False

                    # D. 세로 라인에서 깊이가 3이상인 빈칸 발생시
                    if deepflag:
                        if y > 2 and self.map[y - 1][x] and self.map[y - 2][x] and self.map[y - 3][x]:
                            if (x > 1 and self.map[y - 3][x - 1]) or (x < WIDTH - 1 and self.map[y - 3][x + 1]):
                                d += 1
                                deepflag = False

                    # F. 단면적 숫자 계산
                    if y < HEIGHT - 1 and self.map[y + 1][x] == 0:
                        f += 1
                    if y > 0 and self.map[y - 1][x] == 0:
                        f += 1
                    if x > 0 and self.map[y][x - 1] == 0:
                        f += 1
                    if x < WIDTH - 1 and self.map[y][x + 1] == 0:
                        f += 1

        # C. 높이 표쥰 편차 구하기
        c = np.std(lineheights)

        # E. 갇힌 블록 갯수
        for x in range(WIDTH):

            check_flag = False

            for y in range(HEIGHT):
                if self.map[y][x]:
                    check_flag = True

                if check_flag and not self.map[y][x]:
                    e += 1

        # G. 블록 최대 높이
        g = max(lineheights)

        poten = A * a + B * b + C * c + D * d + E * e + F * f + G * g
        poten = np.round(poten, 3)

        # print('POTEN : {0}  A :{1} B :{2}  C :{3} D :{4} E :{5}  F :{6} '.format(poten, a,b,c,d,e,f))

        self.poten = poten

    def calCombo(self):

        self.combo = pow(self.comboline, 2) * 25

    # def showQ(self, Qtable):
    #
    #     for i in range(len(self.Qboard)):
    #         self.canvas.delete(self.Qboard[i])
    #
    #     self.Qboard = [ [] for _ in range(len(Qtable[0]))]
    #
    #     #print(Qtable[0][10])
    #     for i in range(4):
    #         for j in range(WIDTH):
    #             # UI 표현
    #             self.Qboard[i*10 + j] = self.canvas.create_text( (WIDTH + 7 + j*2)*UNIT, (HEIGHT/5) * (1 + i)*UNIT,
    #                                                       fill="indianred1",
    #                                                       font="Times 13 bold",
    #                                                       text=str( round(Qtable[0][i*10 + j],3) ) )



    # DQN - Agent , Line 번호를 Action 으로 받는다 ## 현재 이용하지 않음
    def step(self, Line_action):

        self.render()

        if self.nowcarrying == False:
            self.block = self.get_cur_block()
            self.nowcarrying = True
            self.reward = 0

        else:   # action 주지 않았을떄는 Down 하게 설정
            reward = self.move(self.convert_action(Line_action))

        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))

    # DQN-Agent 가 게임 진행 이용하는 함수 , Line 번호 + 회전 타입 을 ACTION 으로 받음
    def line_step(self, action):

        # rendering
        if self.DQN_TEST:
            self.render_TEST()
        else:
            self.render()

        if self.pause:
            while(1):
                time.sleep(1)

        if self.nowcarrying == False:
            self.block = self.get_cur_block()
            self.nowcarrying = True
            self.reward = 0

        else:  # action 주지 않았을떄는 Down 하게 설정
            #print("라인스텝중 타입 : ", self.curr_block[1])
            while self.nowcarrying == True:
                #self.render()  # 실제 이동을 보고 싶으면 주석해제
                for __ in range(4):
                    self.move(self.convert_action(action))
                self.move(DOWN)

        self.canvas.itemconfigure(self.score_board,
                                 text=basic_score_str + str(int(self.score)))

    # Agent 의 action 을 실제 게임의 action 으로 전환해주는 함수
    def convert_action(self, action):


        if self.nowcarrying and len(self.block) > 0:
            x = self.block_pos[0]
            nx = int(x/UNIT)
            type = self.curr_block[1]
            r = self.curr_block[2]
        else:
            return 0

        # gx , gr 계산
        gx = 0          # goal x
        gr = 0          # goal r

        if type == 0:               # O 자 블록
            gx = action
            gr = 0

        elif type == 1:             # l 자 블록
            if action < 10:
                gr = 0
                gx = action
            else:
                gr = 1
                gx = action - WIDTH

        elif type == 2 or type == 3:    # J / L 자 블록

            if action < 9:
                gr = 0
                gx = action
            elif action < 17:
                gr = 1
                gx = action - 8
            elif action < 26:
                gr = 2
                gx = action - 16
            elif action < 34:
                gr = 3
                gx = action - 25

        #print("Block type : %d , Action : %d , gx : %d , gr : %d" % (gx, gr))

        if r != gr:    # 회전
            self.Rotate()
            return 0

        elif nx > gx:    # 왼쪽으로 이동
            ret = LEFT
        elif nx < gx:    # 오른쪽으로 이동
            ret = RIGHT
        elif nx == gx:
            ret = 0

        # 현재 기준점 , 원하는 Line , action  출력
        #print("nx : ", nx, "Line : ", line , " ROT : ", r_idx)

        return ret

def pre_processing(curr_map):
    copy_map = copy.deepcopy(curr_map)
    # ny, nx = 4.20, 10.5
    # for n in range(20):
    #     for m in range(8):
    #         for i in range(int(n * ny), int(n * ny + ny)):
    #             for j in range(int(m * nx), int(m * nx + nx)):
    #                 ret[i][j] = copy_map[n][m]
    return copy_map


# 테스트용 메인문 ( 명령 직접 입력 )
if __name__ == "__main__":
    tetris = Tetris()
    for __ in range(10000000000):
        tetris.game_update()



