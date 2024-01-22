import cv2

class MNISTLearner:
    def fit(self, x_train, y_train):
        n, rows, cols = x_train.shape

        images = x_train.copy()
        for i in range(len(images)):
            _, images[i] = cv2.threshold(images[i], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        self.predictor = [[{0: 0,
                            1: 0,
                            2: 0,
                            3: 0,
                            4: 0, 
                            5: 0,
                            6: 0,
                            7: 0,
                            8: 0,
                            9: 0} for x in range(cols)] for y in range(rows)]
        for i in range(n):
            image = images[i]
            for y in range(rows):
                for x in range(cols):
                    if (image[y][x] == 255):
                        self.predictor[y][x][y_train[i]] += 1

    def predict(self, x_test):
        y_pred = []

        images = x_test.copy()
        for i in range(len(images)):
            _, images[i] = cv2.threshold(images[i], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        for image in images:
            y_pred.append(self.predict_single(image))
        return y_pred

    def predict_single(self, test_sample):
        rows, cols = test_sample.shape
        votes = {}
        for i in range(10):
            votes[i] = 0
        for y in range(rows):
            for x in range(cols):
                if (test_sample[y][x] == 255):
                    dict_sum = sum(self.predictor[y][x].values())
                    if (dict_sum != 0):
                        for key, value in self.predictor[y][x].items():
                            votes[key] += value/dict_sum

        return max(votes, key=votes.get)
