import numpy as np
import os
import matplotlib.pyplot as plt

# 시각화 모듈
class Visualizer:
    def __init__(self):

        self.fig = None  # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.axes = None  # 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

        # 시각화 파라미터
        self.num_Update = 1000   # X축 갱신 단위

        # 경로 지정
        self.train_summary_dir = os.path.join('./', 'Train_summary')
        if not os.path.isdir(self.train_summary_dir):
            os.makedirs(self.train_summary_dir)


    # 1M Step 동안 얻은 SCORE 점수를 출력
    def plot_SCORE(self, SCORE):

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("1M Step SCORE")
        ax1.plot(SCORE, label="SCORE")
        #ax1.plot([PV[0] for _ in range(len(PV))], label="Start Value")
        ax1.legend()
        ax1.grid(True)

        # 저장 - 경로 : epoch_summary
        saveFile = os.path.join(
            self.train_summary_dir, 'Score.png')
        plt.savefig(saveFile)

        plt.close(fig)

    # Qmax 출력
    def plot_Qmax(self, Qmax):


        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Avg Qmax")
        ax1.plot(Qmax, label="Qmax")
        #ax1.plot([PV[0] for _ in range(len(PV))], label="Start Value")
        ax1.legend()
        ax1.grid(True)

        # 저장 - 경로 : epoch_summary
        saveFile = os.path.join(
            self.train_summary_dir, 'Qmax.png')
        plt.savefig(saveFile)

        plt.close(fig)

    # Reward 출력
    def plot_Reward(self, Reward):

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Reward")
        ax1.plot(Reward, label="Reward")
        #ax1.plot([PV[0] for _ in range(len(PV))], label="Start Value")
        ax1.legend()
        ax1.grid(True)

        # 저장 - 경로 : epoch_summary
        saveFile = os.path.join(
            self.train_summary_dir, 'Reward.png')
        plt.savefig(saveFile)

        plt.close(fig)

    # Step 출력
    def plot_Steps(self, Steps):

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Steps")
        ax1.plot(Steps, label="Step")
        #ax1.plot([PV[0] for _ in range(len(PV))], label="Start Value")
        ax1.legend()
        ax1.grid(True)

        # 저장 - 경로 : epoch_summary
        saveFile = os.path.join(
            self.train_summary_dir, 'Step.png')
        plt.savefig(saveFile)

        plt.close(fig)

    # Loss 출력
    def plot_Loss(self, Loss):

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Loss")
        ax1.plot(Loss, label="Loss")
        #ax1.plot([PV[0] for _ in range(len(PV))], label="Start Value")
        ax1.legend()
        ax1.grid(True)

        # 저장 - 경로 : epoch_summary
        saveFile = os.path.join(
            self.train_summary_dir, 'Loss.png')
        plt.savefig(saveFile)

        plt.close(fig)
