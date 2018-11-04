import numpy as np
import time
#SJC 14/10/2018

def initall(n, m, bias):
    # n 输入维度，m 训练样本数量
    Xinput = np.random.rand(n, m)
    Weight = np.random.rand(n, 1)
    Bias = bias
    return Xinput, Weight, Bias


def sigmoid(Z):
    sigmoid = 1 / (1 + np.exp(-Z))
    return sigmoid


# Xinput n*m矩阵，Youtput 矩阵
def GradiantD(Xinput, Y, learningRate, Weight, Bias, times, samplenum):
    for i in range(0, times):
        # Z：1*m矩阵 预测值
        Z = np.add(np.dot(Weight.T, Xinput), Bias)
        # A：1*m矩阵 sigmoid激活
        ##########分类模型
        # A = sigmoid(Z)
        # # 损失函数 J = yi*log(ai)+(1-yi)log(1-ai)  yi=1时，dJ/dw =dJ/da * da/dy * dy/dw = 1/ai * ai(1-ai) * X = (y-a)*X; yi=0时， dJ/dw= -1/(1-ai) * ai(1-ai) *X = 0-ai*X = (y-a)*X
        # dZ = Y - A
        ##########线性模型
        dZ = Y - Z
        # dJ/dW 和 dJ/dB, n*1 和实数
        dW = (1 / samplenum) * np.dot(Xinput, dZ.T)
        dB = (1 / samplenum) * np.sum(dZ)
        # 优化
        Weight = Weight + learningRate * dW
        Bias = Bias + learningRate * dB
        if (times%100==0):
            print(Weight.T)
    return Weight, Bias


if __name__ == '__main__':
    # 初始化输入，权值和bias
    Xinput, Weight, Bias = initall(2, 1000, 1)
    # 模型定义 y=5Xa+3Xb+0.5
    model = np.array([5, 3])
    Y = np.add(np.dot(model, Xinput), 0.5)
    finalW, finalB = GradiantD(Xinput, Y, 0.005, Weight, Bias, 100000, 1000)
    print(finalW.T)
    print(finalB)
