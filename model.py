import numpy as np

class Model:
    def __init__(self, X, y, epochs = 20, learning_rate = 0.1, w = 2):
        # 파라메터
        self.weights = np.random.rand(w)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.inputs = X
        self.outputs = y

    def train(self):
        for epoch in range(self.epochs):
            for i in range(len(self.inputs)):
                # 총 입력 계산
                total_input = np.dot(self.inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = self.outputs[i] - prediction
                print(f'inputs[i] : {self.inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += self.learning_rate * error * self.inputs[i]
                self.bias += self.learning_rate * error
                print('====')

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)
    