
import time
import numpy as np
import tkinter as tk
#from PIL import ImageTk , Image


#hotoImage = ImageTk.PhotoImage # 이미지
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

## HURISTIC AI (테트리스 게임 데이터 내부 포함)  ##

# gameloop 설정이나 delete 키 이용하여 수동 조작도 가능

class HAI(tk.Tk) :

    def __init__(self):
        super(HAI, self).__init__()

        # 게임 신호 관련
        self.clk = 0
        self.nowcarrying = False  # 블록 이동 시그널
        self.Np = [MID, 0]  # 현재 블록 위치
        self.gameover = False # 게임 종료 시그널
        self.pause = False      # 중지 시그널

        # 점수 관련
        self.score = 0.0
        self.counter = 0
        self.reward = 0
        self.poten = 0

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

        self.block = []
        self.block_pos = []

        self.curr_block = [[],[],[]]
        self.curr_block[0] = -1  # 현재블록 타입 ( 색깔 )
        self.curr_block[1] = -1
        self.curr_block[2] = 0

        ## 인공지능 관련
        self.refblock = []          # 블록기준좌표 보여주는 블록
        self.guideblock = []        # 최적의 위치를 그려주는 블록
        self.potenboards = []       # 놓을 수 있는 위치에 따라 UI 표기용 블록   
        self.goalpos = [0, 0, 0]    # 현재 블록을 놓을 위치

        # 우회 경로 (DFS)
        self.wp = []                # 경로 저장
        self.Opt_wp = []
        self.min_count = 100


        # 맵 정보
        self.canvas, self.counter_board, self.score_board, self.potenboard = self._build_canvas()


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
        #self.game_loop()       # 수동 키
        self.AI_loop()          # HAI




    ## 캔버스 생성단 - tk 라이브러리 이용
    def _build_canvas(self):

        # 공간 분할 #
        canvas = tk.Canvas(self, bg='black',
                           height=HEIGHT * UNIT,
                           width= (WIDTH + 13) * UNIT)
        counter_board = canvas.create_text((WIDTH + 3) * UNIT, int(HEIGHT / 4 * UNIT),
                                           fill="white",
                                           font="Times 10 bold",
                                           text=basic_counter_str + str(int(self.counter)))

        score_board = canvas.create_text((WIDTH + 3) * UNIT, int(HEIGHT / 2 * UNIT),
                                         fill="white",
                                         font="Times 10 bold",
                                         text=basic_score_str + str(int(self.score)))

        poten_board = canvas.create_text((WIDTH + 3) * UNIT, int(HEIGHT / 4 * 3 * UNIT),
                                              fill="white",
                                              font="Times 10 bold",
                                              text="poten score" + str(int(self.poten)))

        # 그리드 생성
        for c in range(0, (WIDTH + 1) * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill='white')
        '''
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)
        '''
        return canvas, counter_board, score_board, poten_board


    ### 블록 생성단  ###

    # 블록 발생 함수 : 블록 이동이 끝났으면 무작위 블록 생성
    def get_cur_block(self):

        # 색깔과 타입 무작위 생성 #
        if self.curr_block[1] == -1:
            self.curr_block[1] = 0
        elif self.curr_block[1] < 6:
            self.curr_block[1] = self.curr_block[1]+1  # 랜덤 블록 타입
        else:
            self.curr_block[1] = 0

        self.curr_block[0] = self.curr_block[1]  # 블록타입에 따라 색이 같음
        self.curr_block[2] = 0  # 회전 타입 ( 디폴트 0 )

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
                self.add_Canvas()

                # 라인체크 -> 라인지울수있으면 지운다
                if self.check_line().__len__() != 0:
                    cl = self.check_line()
                    self.reward = self.clear_line(cl)  # 라인지우기 , REWARD 발생부분
                    time.sleep(0.1)
                    self.map = self.down_all_canvas(cl)  # 라인지웟으면 전체블록 이동

                # 게임종료체크
                if self.is_game_end():
                    #self.reset()
                    self.reward = -20

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

        # poten 계산 및 보여주기 (회전 할 때 마다 변하므로 Rotate 함수에 들어가있음)
        for i in range(len(self.potenboards)):
            self.canvas.delete(self.potenboards[i])
        self.potenboards = self.showPotential()

    # Drop 함수
    def hard_drop(self):

        if self.nowcarrying == False:
            return 0

        while self.possible_move(0) != 2:
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

        reward = 0
        for crect in self.canvas.find_withtag("rect"):
            ny = int(self.canvas.coords(crect)[1] / UNIT)
            if cl.count(ny) != 0:
                nx = int(self.canvas.coords(crect)[0] / UNIT)
                self.map[ny][nx] = 0
                self.canvas.delete(crect)

        for y in cl:
            reward = reward + y - 1
        # print("reward :" , reward)
        self.score += reward

        return reward

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
        for n in range(3):
            for m in range(WIDTH):
                if self.map[n][m] == 1:
                    return True
        return False

    # 리셋 함수
    def reset(self):

        self.curr_block[1] = -1
        self.score = 0.0
        self.reward = 0
        self.counter += 1
        self.canvas.itemconfigure(self.counter_board,
                                  text=basic_counter_str + str(int(self.counter)))
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))
        self.update()
        self.canvas.delete("rect")
        self.nowcarrying = False
        self._clear_map()
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
        time.sleep(0.1)    #학습시 주석처리 , 실제 게임 진행 보고싶으면 0.1 ~ 0.3 설정

    # 메인 업데이트 함수
    def game_update(self):

        # render 함수
        self.render()

        # canvas 업데이트
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))
        self.canvas.itemconfigure(self.potenboard,
                                  text="potenial : " + str(float(self.poten)))

    # 수동 조작용 루프
    def game_loop(self):

        self.clk = self.clk + 1                     # 게임 내 클락

        if self.nowcarrying == False:
            self.block = self.get_cur_block()
            self.nowcarrying = True

        else:
            # 참고용으로 Ref 블록 생성하게 해놓았따
            self.canvas.delete(self.refblock)
            self.refblock = self.printRefBlock()

            # 3 clk 에 한번 자동 이동
            if self.clk % 3 == 0:
                self.move(0)

        if self.clk > 10000:
            self.clk = 0

        self.after(300, self.game_loop)         # 재귀 루프


    ##========================##
    ### HURISTIC AI 추가 부분 ###
    ##========================##

    ## 실제 루프 / 스텝 함수 ##

    # HURISTIC AI 루프
    def AI_loop(self):


        if self.nowcarrying == False:
            self.block = self.get_cur_block()       # 블록 생성

            ## FOP 및 가이드 시각화

            for i in range(len(self.guideblock)):
                self.canvas.delete(self.guideblock[i])
            self.guideblock = []

            self.poten = self.calPotential()

            # poten 계산 및 보여주기
            for i in range(len(self.potenboards)):
                self.canvas.delete(self.potenboards[i])
            self.potenboards = self.showPotential()

            # drawOptimalPos
            self.canvas.delete(self.guideblock)
            self.guideblock = self.drawGuide()

            self.goalpos = self.FindOptimalPos()

            self.nowcarrying = True

        else:
            # 참고용으로 Ref 블록 생성하게 해놓았따
            self.canvas.delete(self.refblock)
            self.refblock = self.printRefBlock()

            if self.is_game_end():
                self.reset()

            else:
                if not self.pause:
                    self.AI_step()  # HAI 의 경우 STEP 함수에 의해 움직인다
                    self.move(0)

        self.after(300, self.AI_loop)

    # HAI 가  목표까지 블록을 이동시키게 하는 스텝 함수
    def AI_step(self):

        # FOP 를 통해 r, x, y 를 받고
        r = self.goalpos[0]
        x = self.goalpos[1]
        y = self.goalpos[2]

        # FOP 가 위치를 못 찾은 경우 (죽은 경우)
        if r == -1:
            return

        # 회전 타입을 맞춘 후
        while self.curr_block[2] != r:
            self.Rotate()

            print(r, self.curr_block[2])

        # 우회 해야 하는지 보고 우회해야 할 경우 경로 불러 온다
        if self.Opt_wp != None:
            self.Opt_wp.reverse()

            for i in range(len(self.Opt_wp)):
                self.update()
                time.sleep(0.1)         # 우선 clk 으로 동기화는 안 시킴
                self.move(self.Opt_wp[i])

            self.Opt_wp = None

        # 우회가 아닌 DROP 의 경우
        else:
            # 동일 STEP 내에서 왼/오른쪽 세번까지 행동 선택 , 만약 넘어가면 다음 스텝 때 진행

            for i in range(5):

                if (self.block_pos[0] / UNIT) > x:
                    self.move(LEFT)
                elif (self.block_pos[0] / UNIT) < x:
                    self.move(RIGHT)
                else:
                    break

            if self.block_pos[0] / UNIT == x:
                self.hard_drop()        # DROP


    ## 최적 위치를 구하기 위한 함수들 ##

    # 현재 블록의 최적위치 탐색 리턴 ( [r,x,y] )
    def FindOptimalPos(self):

        b = self.curr_block[0]

        # 초기화
        MAX_POTEN = -100
        tmp = 0
        blocks = None
        Opt_X = -1
        Opt_Y = -1
        Opt_R = -1

        # 우회 해야할 경우 경로를 기억
        self.Opt_wp = None

        # 4가지 회전타입과 각 세로 라인에 대해 최적의 점수를 계산함
        for rot in range(4):
            for x in range(WIDTH):
                for y in range(HEIGHT):
                    self.wp = None
                    blocks = self.checkValid_Blockpos([b, rot, x, y])
                    if blocks != None:
                        tmp = self.block_simulate(blocks)
                        if tmp >= MAX_POTEN:
                            MAX_POTEN = tmp
                            Opt_X = x
                            Opt_Y = y
                            Opt_R = rot

                            # 우회 경로 일 경우에는 경로 저장
                            if self.wp != None:
                                self.Opt_wp = self.wp
                            else:
                                self.Opt_wp = None

        return [Opt_R, Opt_X, Opt_Y]

    # 현재 게임에서 상태 점수 계산
    def calPotential(self):

        # 어드밴티지 요소 초기화
        a = 0       # a 는 현재 쌓인 블록의 숫자
        b = 0       # b 는 라인클리어

        # 패널티 요소 초기화
        c = 0       # c 는 평균 높이에 따른 편차
        d = 0       # d 는 3칸이상 세로 구덩이 숫자
        e = 0       # e 는 세로줄 기준으로 빈칸 숫자
        f = 0       # f 는 쌓인 블록에 대한 단면적
        g = 0       # g 는 최대 높이

        # C 구하기 위한 임시 변수
        lineheights = [0 for _ in range(WIDTH)]

        for x in range(WIDTH):
            lineflag = True          # 라인별 최대 높이를 구하기 위한 flag
            deepflag = True

            for y in range(HEIGHT):

                # A. 전체 블록의 갯수 구하기
                if self.map[y][x] == 1:
                    a += 1

                    # C 계산을 위해 세로 라인별 높이 저장
                    if lineflag :
                        lineheights[x] = (-y + HEIGHT)
                        lineflag = False

                    # D. 세로 라인에서 깊이가 3이상인 빈칸 발생시
                    if deepflag:
                        if y > 2 and self.map[y - 1][x] and self.map[y - 2][x] and self.map[y - 3][x]:
                            if (x > 1 and self.map[y - 3][x - 1]) or (x < WIDTH - 1 and self.map[y - 3][x + 1]):
                                d += 1
                                deepflag = False

                    # F. 단면적 숫자 계산
                    if y < HEIGHT-1 and self.map[y+1][x] == 0:
                        f +=1
                    if y > 0 and self.map[y-1][x] == 0:
                        f += 1
                    if x > 0 and self.map[y][x-1] == 0:
                        f += 1
                    if x < WIDTH-1 and self.map[y][x+1] == 0:
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

        poten = A*a + B*b + C*c + D*d + E*e + F*f + G*g
        poten = np.round(poten, 3)

        #print('POTEN : {0}  A :{1} B :{2}  C :{3} D :{4} E :{5}  F :{6} '.format(poten, a,b,c,d,e,f))

        return poten

    # 우회경로 - DFS 방식으로 탐색
    def SearchDFS(self, b, r, np, gp, count, waypoint):

        if count > self.min_count:
            return False

        if count > 10:
            return False

        # 도착지점 도착했는지 or count 확인
        if np[0] == gp[0] and np[1] == gp[1]:
            self.wp = waypoint
            print("경로찾았으", count)
            if self.min_count > count:
                self.min_count = count
            return True


        # 다음 유효한 행동 탐색 : 경로 탐색시 도착 -> 시작 으로 탐색하므로 모든 행동을 바꾸어서 저장
        if self.checkValidDir(b, r, np, LEFT):
            self.SearchDFS(b, r, [np[0]-1, np[1]], gp, count + 1, waypoint + [RIGHT])

        if self.checkValidDir(b, r, np, RIGHT):
            self.SearchDFS(b, r, [np[0] + 1, np[1]], gp, count + 1, waypoint + [LEFT])

        idx = 0
        if self.checkValidDir(b, r, np, DOWN):
            while self.checkValidDir(b, r, [np[0], np[1] - idx ], DOWN):
                idx = idx + 1
            self.SearchDFS(b, r, [np[0], np[1] - idx ], gp, count + 1, waypoint + [DOWN for _ in range(idx)])

    # DFS 방식 - 방향 유효한지 판단
    def checkValidDir(self, b, r, pos, dir):

        x = pos[0]
        y = pos[1]

        if dir == LEFT:
            # 놓아질 블록 위치 좌표 계산
            pos1 = [x + blocks_dx[b][r][0] - 1, y + blocks_dy[b][r][0]]
            pos2 = [x + blocks_dx[b][r][1] - 1, y + blocks_dy[b][r][1]]
            pos3 = [x + blocks_dx[b][r][2] - 1, y + blocks_dy[b][r][2]]
            pos4 = [x + blocks_dx[b][r][3] - 1, y + blocks_dy[b][r][3]]
            posN = [pos1, pos2, pos3, pos4]

        elif dir == RIGHT:
            # 놓아질 블록 위치 좌표 계산
            pos1 = [x + blocks_dx[b][r][0] + 1, y + blocks_dy[b][r][0]]
            pos2 = [x + blocks_dx[b][r][1] + 1, y + blocks_dy[b][r][1]]
            pos3 = [x + blocks_dx[b][r][2] + 1, y + blocks_dy[b][r][2]]
            pos4 = [x + blocks_dx[b][r][3] + 1, y + blocks_dy[b][r][3]]
            posN = [pos1, pos2, pos3, pos4]

        elif dir == DOWN:       # DOWN 의 경우 역산하므로 UP 으로 연산
            # 놓아질 블록 위치 좌표 계산
            pos1 = [x + blocks_dx[b][r][0], y + blocks_dy[b][r][0] - 1]
            pos2 = [x + blocks_dx[b][r][1], y + blocks_dy[b][r][1] - 1]
            pos3 = [x + blocks_dx[b][r][2], y + blocks_dy[b][r][2] - 1]
            pos4 = [x + blocks_dx[b][r][3], y + blocks_dy[b][r][3] - 1]
            posN = [pos1, pos2, pos3, pos4]


        # 블록의 X 좌표가 범위를 넘어가는 경우는 놓을 수 없음
        if (pos1[0] < 0 or pos1[0] >= WIDTH) or (pos2[0] < 0 or pos2[0] >= WIDTH) or (
                pos3[0] < 0 or pos3[0] >= WIDTH) or (pos4[0] < 0 or pos4[0] >= WIDTH):
            return False

        # 블록이 Y 좌표를 넘어가는 경우는 놓을 수 없음
        if pos1[1] > HEIGHT - 1 or pos2[1] > HEIGHT - 1 or pos3[1] > HEIGHT - 1 or pos4[1] > HEIGHT - 1:
            return False
        if pos1[1] < 0 or pos2[1] < 0 or pos3[1]< 0 or pos4[1] < 0 :
            return False


        # 블록 이동이 기존 블록에 충돌하는 경우
        if (not self.map[pos1[1]][pos1[0]] and not self.map[pos2[1]][pos2[0]] and
            not self.map[pos3[1]][pos3[0]] and not self.map[pos4[1]][pos4[0]]):
            return True

    # 블록을 놓을 수 있는 유효한 위치인지 검사 ( blocks = [b,r,x,y] )
    def checkValid_Blockpos(self, blocks):

        # 블록 타입 , 회전 타입 , 기준 좌표 x,y
        b = blocks[0]
        r = blocks[1]
        x = blocks[2]
        y = blocks[3]

        # 놓아질 블록 위치 좌표 계산
        pos1 = [x + blocks_dx[b][r][0], y + blocks_dy[b][r][0]]
        pos2 = [x +blocks_dx[b][r][1], y + blocks_dy[b][r][1]]
        pos3 = [x + blocks_dx[b][r][2], y + blocks_dy[b][r][2]]
        pos4 = [x + blocks_dx[b][r][3], y + blocks_dy[b][r][3]]
        posN = [pos1, pos2, pos3, pos4]

        # 블록의 X 좌표가 범위를 넘어가는 경우는 놓을 수 없음
        if (pos1[0] < 0 or pos1[0] >= WIDTH) or (pos2[0] < 0 or pos2[0] >= WIDTH) or (
                pos3[0] < 0 or pos3[0] >= WIDTH) or (pos4[0] < 0 or pos4[0] >= WIDTH):
            return None

        # 블록이 Y 좌표를 넘어가는 경우는 놓을 수 없음
        if pos1[1] > HEIGHT - 1 or pos2[1] > HEIGHT - 1 or pos3[1] > HEIGHT - 1 or pos4[1] > HEIGHT - 1:
            return None

        # 1.블록이 벽/블록에 붙엇는지 체크
        if ( pos1[1] == HEIGHT - 1 or self.map[pos1[1] + 1][pos1[0]] or
             pos2[1] == HEIGHT - 1 or self.map[pos2[1] + 1][pos2[0]] or
             pos3[1] == HEIGHT - 1 or self.map[pos3[1] + 1][pos3[0]] or
             pos4[1] == HEIGHT - 1 or self.map[pos4[1] + 1][pos4[0]] ) :

            # 2.블록을 놓을 수 있는지 체크
            if (not self.map[pos1[1]][pos1[0]] and not self.map[pos2[1]][pos2[0]] and
                    not self.map[pos3[1]][pos3[0]] and not self.map[pos4[1]][pos4[0]]):
                pass
            else:
                return None

            #3.블록을 실제 이동시킬 수 있는지 체크

            # 3-1. HardDrop 가능한지 체크
            Drop_flag = True
            Bypass_flag = True
            ref_block = []

            for ref_y in range(y):
                ref_pos = [x, ref_y]
                ref_pos1 = [ref_pos[0] + blocks_dx[b][r][0], ref_pos[1] + blocks_dy[b][r][0]]
                ref_pos2 = [ref_pos[0] + blocks_dx[b][r][1], ref_pos[1] + blocks_dy[b][r][1]]
                ref_pos3 = [ref_pos[0] + blocks_dx[b][r][2], ref_pos[1] + blocks_dy[b][r][2]]
                ref_pos4 = [ref_pos[0] + blocks_dx[b][r][3], ref_pos[1] + blocks_dy[b][r][3]]
                ref_block = [ref_pos1, ref_pos2, ref_pos3, ref_pos4]

                # Drop 이 불가시
                if (self.map[ref_pos1[1]][ref_pos1[0]] or self.map[ref_pos2[1]][ref_pos2[0]] or
                        self.map[ref_pos3[1]][ref_pos3[0]] or self.map[ref_pos4[1]][ref_pos4[0]]):
                    Drop_flag = False

            if Drop_flag:
                return posN

            # 3-2. 우회 이동이 가능한지 체크
            self.min_count = 100                                # DFS 서치 전 최소카운트 초기화
            self.SearchDFS(b, r, [x,y], [MID/UNIT, 0], 0 , [])

            if self.wp != None:
                return posN

        #         # 3-3. 스핀으로 끼워넣기 가능한지 체크 (아직 미구현)
        #         pass
        # pass

    # 해당 좌표에 블록 놓는 시뮬레이션
    def block_simulate(self, ref_pos):

        # 현재블록을 ref_pos 에 맞춰 배치시켰다고 가정할 경우 POTEN 계산
        for block in range(len(ref_pos)):
            self.map[ ref_pos[block][1] ][ ref_pos[block][0] ] = 1

        expect = self.calPotential()

        # 실제 블록을 이동시키지 않았으므로 복구 시킴
        for block in range(len(ref_pos)):
            self.map[ ref_pos[block][1] ][ ref_pos[block][0] ] = 0

        return expect


    ## 피드백을 위해 UI로 출력 해주는 함수 ##

    # 최적의 블록위치를 받아 보여주도록 : GRAY85
    def drawGuide(self):

        color = "gray95"

        r, x, y = self.FindOptimalPos()
        x = UNIT*x
        y = UNIT*y

        b = self.curr_block[1]

        rect1 = self.canvas.create_rectangle(x + UNITblocks_dx[b][r][0] , y + UNITblocks_dy[b][r][0],
                                             x + UNITblocks_dx[b][r][0] +UNIT, y + UNITblocks_dy[b][r][0] + UNIT,
                                             fill= color, tag="rect")
        rect2 = self.canvas.create_rectangle(x + UNITblocks_dx[b][r][1] , y + UNITblocks_dy[b][r][1],
                                             x + UNITblocks_dx[b][r][1] +UNIT, y +UNITblocks_dy[b][r][1] + UNIT,
                                             fill= color, tag="rect")
        rect3 = self.canvas.create_rectangle(x + UNITblocks_dx[b][r][2] , y + UNITblocks_dy[b][r][2],
                                             x + UNITblocks_dx[b][r][2] +UNIT, y + UNITblocks_dy[b][r][2] + UNIT,
                                             fill= color, tag="rect")
        rect4 = self.canvas.create_rectangle(x + UNITblocks_dx[b][r][3] , y + UNITblocks_dy[b][r][3],
                                             x + UNITblocks_dx[b][r][3] +UNIT, y + UNITblocks_dy[b][r][3] + UNIT,
                                             fill= color, tag="rect")
        block = [rect1, rect2, rect3, rect4]

        return block

    # 현재 블록을 놓을 수 있는 좌표의 점수들 가시화
    def showPotential(self):

        # 가능한 좌표
        possible_pos = []

        # 가능한 좌표에 따른 포텐 점수들
        possible_potens = []

        # 현재 블록의
        b = self.curr_block[0]
        r = self.curr_block[2]

        for x in range(WIDTH):
            for y in range(HEIGHT):
                blocks = self.checkValid_Blockpos([b,r,x,y])

                if blocks != None:
                    possible_pos.append([x,y])
                    possible_potens.append(self.block_simulate(blocks))

        # 전부 검사하였다면 캔버스에 숫자 출력하게 끔

        poten_boards = [ [] for _ in range(len(possible_pos))]

        for i in range(len(possible_pos)):
            # UI 표현
            poten_boards[i] = self.canvas.create_text(possible_pos[i][0] * UNIT + 0.5*UNIT, int(possible_pos[i][1] * UNIT) + 0.5*UNIT,
                                               fill="indianred1",
                                               font="Times 15 bold",
                                               text= str(int(possible_potens[i])) )
        return poten_boards

    # showPotential 가독성을 위해 추가
    def printRefBlock(self):

        pos = self.block_pos

        ref_pos = self.canvas.create_text(pos[0] +0.5*UNIT , pos[1] + 0.5*UNIT,
                                             fill= "lightcyan1", font="Times 15 bold", text = "Rf")
        return ref_pos




    ##========================##
    ### DQN AI 추가 부분 ###
    ##========================##

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

        self.render()

        if self.nowcarrying == False:
            self.block = self.get_cur_block()
            self.nowcarrying = True
            self.reward = 0

        else:  # action 주지 않았을떄는 Down 하게 설정
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
            r = self.curr_block[2]
        else:
            return 0

        line = action % 8
        r_idx = action // 8

        if r != r_idx:    # 회전
            self.Rotate()
            return 0
        elif nx > line:    # 왼쪽으로 이동
            ret = LEFT
        elif nx < line:    # 오른쪽으로 이동
            ret = RIGHT
        elif nx == line:
            ret = 0

        # 현재 기준점 , 원하는 Line , action  출력
        #print("nx : ", nx, "Line : ", action_line , " Action : ", ret)

        return ret


# 테스트용 메인문 ( 명령 직접 입력 )
if __name__ == "__main__":
    tetris = HAI()
    for __ in range(1000000):
        tetris.game_update()



