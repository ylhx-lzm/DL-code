# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        """
        计算神经网络的梯度。
        
        参数:
        x -- 输入数据
        t -- 目标标签
        
        返回:
        grads -- 包含权重和偏置的梯度字典
        """
        # 获取权重和偏置参数
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        # 初始化梯度字典
        grads = {}
        
        # 计算批次数量
        batch_num = x.shape[0]
        
        # forward前向传播
        a1 = np.dot(x, W1) + b1  #当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        #批量的总损失函数L是所有样本损失的平均
        
        # backward反向传播
        da2 = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, da2) # 公式: ∂L/∂W2 = z1.T *(y - t) / batch_size
        z1 = np.dot(da2, W2.T)  # 公式: ∂L/∂z1 = ∂L/∂a2 * ∂a2/∂z1 = da2 * W2.T
        grads['b2'] = np.sum(da2, axis=0)
        #每个分量的梯度是所有样本在该分量上的梯度之和
        
        dz1 = np.dot(da2, W2.T)  #
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
    
        return grads