import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


class MarkovChain():
    def __init__(self, mode='single'):
        self.mode = mode
        self.A = None
        self.pi = None
        self.alfabet = []

    def fit(self, X):
        if self.mode == 'single':
            for sentence in X:
                words = nltk.word_tokenize(sentence, language="polish")
                for word in words:
                    for letter in word:
                        if letter not in self.alfabet:
                            self.alfabet.append(letter)
        elif self.mode == 'double':
            tmp = ""
            for sentence in X:
                words = nltk.word_tokenize(sentence, language="polish")
                for word in words:
                    for letter in word:
                        if len(tmp) < 2:
                            tmp += letter
                        else:
                            if tmp not in self.alfabet:
                                self.alfabet.append(tmp)
                            tmp = ""
        elif self.mode == 'word':
            pass

        self.alfabet = np.array(self.alfabet)

        self.A = np.zeros((len(self.alfabet), len(self.alfabet)))
        self.pi = np.zeros(len(self.alfabet))
        if self.mode == 'single':
            for sentence in X:
                words = nltk.word_tokenize(sentence, language="polish")
                piIdx = np.where(self.alfabet == words[0][0])[0]
                self.pi[piIdx] += 1
                lastLetter = None
                for i, word in enumerate(words):
                    for j, letter in enumerate(word):
                        if i == 0 and j == 0:
                            lastLetter = letter
                            continue
                        AIdxx = np.where(self.alfabet == lastLetter)[0]
                        AIdxy = np.where(self.alfabet == letter)[0]
                        self.A[AIdxx, AIdxy] += 1
                        lastLetter = letter
        elif self.mode == 'double':
            for sentence in X:
                words = nltk.word_tokenize(sentence, language="polish")
                piIdx = np.where(self.alfabet == words[0][0:2])[0]
                self.pi[piIdx] += 1
                lastLetter = ""
                currentLetter = ""
                for i, word in enumerate(words):
                    for j, letter in enumerate(word):
                        if i == 0 and (j == 0 or j == 1):
                            lastLetter += letter
                            continue
                        if len(currentLetter) < 2:
                            currentLetter += letter
                            continue
                        AIdxx = np.where(self.alfabet == lastLetter)[0]
                        AIdxy = np.where(self.alfabet == currentLetter)[0]
                        self.A[AIdxx, AIdxy] += 1
                        lastLetter = currentLetter
                        currentLetter = ""
        elif self.mode == 'word':
            pass
        # Poprawka Laplace'a
        self.A = self.A + 1 / (5 * self.A.shape[0])
        for i in range(self.A.shape[0]):
            self.A[i, :] = self.A[i, :] / np.sum(self.A[i, :])
        self.pi = self.pi / np.sum(self.pi)

    def predict(self, X):
        predictions = np.ones(len(X))
        if self.mode == 'single':
            for k, sentence in enumerate(X):
                words = nltk.word_tokenize(sentence, language="polish")
                piIdx = np.where(self.alfabet == words[0][0])[0]
                if self.pi[piIdx].size > 0:
                    predictions[k] *= self.pi[piIdx]
                lastLetter = None
                for i, word in enumerate(words):
                    for j, letter in enumerate(word):
                        if i == 0 and j == 0:
                            lastLetter = letter
                            continue
                        AIdxx = np.where(self.alfabet == lastLetter)[0]
                        AIdxy = np.where(self.alfabet == letter)[0]
                        if self.A[AIdxx, AIdxy].size == 0:
                            lastLetter = letter
                            continue
                        predictions[k] *= self.A[AIdxx, AIdxy]
                        lastLetter = letter
        elif self.mode == 'double':
            for k, sentence in enumerate(X):
                words = nltk.word_tokenize(sentence, language="polish")
                piIdx = np.where(self.alfabet == words[0][0:2])[0]
                if self.pi[piIdx].size > 0:
                    predictions[k] *= self.pi[piIdx]
                lastLetter = ""
                currentLetter = ""
                for i, word in enumerate(words):
                    for j, letter in enumerate(word):
                        if i == 0 and (j == 0 or j == 1):
                            lastLetter += letter
                            continue
                        if len(currentLetter) < 2:
                            currentLetter += letter
                            continue
                        AIdxx = np.where(self.alfabet == lastLetter)[0]
                        AIdxy = np.where(self.alfabet == currentLetter)[0]
                        if self.A[AIdxx, AIdxy].size == 0:
                            lastLetter = currentLetter
                            continue
                        predictions[k] *= self.A[AIdxx, AIdxy]
                        lastLetter = currentLetter
                        currentLetter = ""
        elif self.mode == 'word':
            pass
        return predictions


def loadWholeExpose(filename):
    file = open(filename, mode='r', encoding='utf8')
    data = file.read()
    file.close()
    return data


def preprocessing(text):
    text = text.lower()
    text = nltk.sent_tokenize(text, language="polish")
    sentences = []
    for sentence in text:
        sentences.append(re.sub("[^a-zA-Ząćęłńóśźż ]+", "", sentence))
    X_train, X_test = train_test_split(sentences, test_size=0.3)
    return X_train, X_test


def makeChains(fname, a_mode):
    expose = loadWholeExpose(rootPath + fname + '.txt')
    X_train, X_test = preprocessing(expose)
    model = MarkovChain(mode=a_mode)
    model.fit(X_train)
    return model, X_test


def makeLabels(test, idx, probabilities, labels):
    predictions = np.empty_like(test)
    for i in range(len(test)):
        tmp = np.array(
            [probabilities[0, idx][i], probabilities[1, idx][i], probabilities[2, idx][i], probabilities[3, idx][i]])
        labelIdx = np.argmax(tmp)
        predictions[i] = labels[labelIdx]
    score = len(np.where(predictions == labels[idx])[0]) / len(test)
    counts = np.unique(predictions, return_counts=True)
    return predictions, score, counts


if __name__ == "__main__":
    rootPath = './data/'
    labels = ['morawiecki', 'tusk', 'szydlo', 'kopacz']
    # morawieckiModel, morawieckiTest = makeChains(labels[0], a_mode='single')
    # tuskModel, tuskTest = makeChains(labels[1], a_mode='single')
    # szydloModel, szydloTest = makeChains(labels[2], a_mode='single')
    # kopaczModel, kopaczTest = makeChains(labels[3], a_mode='single')
    # # Testy
    # probabilities = np.empty((len(labels), len(labels)), dtype=object)
    # # [i, j] i-model, j-test
    # probabilities[0, 0] = morawieckiModel.predict(morawieckiTest)
    # probabilities[0, 1] = morawieckiModel.predict(tuskTest)
    # probabilities[0, 2] = morawieckiModel.predict(szydloTest)
    # probabilities[0, 3] = morawieckiModel.predict(kopaczTest)

    # probabilities[1, 0] = tuskModel.predict(morawieckiTest)
    # probabilities[1, 1] = tuskModel.predict(tuskTest)
    # probabilities[1, 2] = tuskModel.predict(szydloTest)
    # probabilities[1, 3] = tuskModel.predict(kopaczTest)

    # probabilities[2, 0] = szydloModel.predict(morawieckiTest)
    # probabilities[2, 1] = szydloModel.predict(tuskTest)
    # probabilities[2, 2] = szydloModel.predict(szydloTest)
    # probabilities[2, 3] = szydloModel.predict(kopaczTest)

    # probabilities[3, 0] = kopaczModel.predict(morawieckiTest)
    # probabilities[3, 1] = kopaczModel.predict(tuskTest)
    # probabilities[3, 2] = kopaczModel.predict(szydloTest)
    # probabilities[3, 3] = kopaczModel.predict(kopaczTest)

    # print("Test tekstu Morawieckiego")
    # morawieckiPredictions, morawieckiScore, morawieckiPomylki = makeLabels(morawieckiTest, 0, probabilities, labels)
    # print(f'Wynik łańcucha Morawieckiego: {morawieckiScore}')
    # print(f'Był mylony z: {morawieckiPomylki}')

    # print("Test tekstu Tuska")
    # tuskPredictions, tuskScore, tuskPomylki = makeLabels(tuskTest, 1, probabilities, labels)
    # print(f'Wynik łańcucha Tuska: {tuskScore}')
    # print(f'Był mylony z: {tuskPomylki}')

    # print("Test tekstu Szydło")
    # szydloPredictions, szydloScore, szydloPomylki = makeLabels(szydloTest, 2, probabilities, labels)
    # print(f'Wynik łańcucha Szydło: {szydloScore}')
    # print(f'Był mylony z: {szydloPomylki}')

    # print("Test tekstu Kopacz")
    # kopaczPredictions, kopaczScore, kopaczPomylki = makeLabels(kopaczTest, 3, probabilities, labels)
    # print(f'Wynik łańcucha Kopacz: {kopaczScore}')
    # print(f'Był mylony z: {kopaczPomylki}')

    # morawieckiModel, morawieckiTest = makeChains(labels[0], a_mode='double')
    # tuskModel, tuskTest = makeChains(labels[1], a_mode='double')
    # szydloModel, szydloTest = makeChains(labels[2], a_mode='double')
    # kopaczModel, kopaczTest = makeChains(labels[3], a_mode='double')
    # # Testy
    # probabilities = np.empty((len(labels), len(labels)), dtype=object)
    # # [i, j] i-model, j-test
    # probabilities[0, 0] = morawieckiModel.predict(morawieckiTest)
    # probabilities[0, 1] = morawieckiModel.predict(tuskTest)
    # probabilities[0, 2] = morawieckiModel.predict(szydloTest)
    # probabilities[0, 3] = morawieckiModel.predict(kopaczTest)
    #
    # probabilities[1, 0] = tuskModel.predict(morawieckiTest)
    # probabilities[1, 1] = tuskModel.predict(tuskTest)
    # probabilities[1, 2] = tuskModel.predict(szydloTest)
    # probabilities[1, 3] = tuskModel.predict(kopaczTest)
    #
    # probabilities[2, 0] = szydloModel.predict(morawieckiTest)
    # probabilities[2, 1] = szydloModel.predict(tuskTest)
    # probabilities[2, 2] = szydloModel.predict(szydloTest)
    # probabilities[2, 3] = szydloModel.predict(kopaczTest)
    #
    # probabilities[3, 0] = kopaczModel.predict(morawieckiTest)
    # probabilities[3, 1] = kopaczModel.predict(tuskTest)
    # probabilities[3, 2] = kopaczModel.predict(szydloTest)
    # probabilities[3, 3] = kopaczModel.predict(kopaczTest)
    #
    # print("Test tekstu Morawieckiego")
    # morawieckiPredictions, morawieckiScore, morawieckiPomylki = makeLabels(morawieckiTest, 0, probabilities, labels)
    # print(f'Wynik łańcucha Morawieckiego: {morawieckiScore}')
    # print(f'Był mylony z: {morawieckiPomylki}')
    #
    # print("Test tekstu Tuska")
    # tuskPredictions, tuskScore, tuskPomylki = makeLabels(tuskTest, 1, probabilities, labels)
    # print(f'Wynik łańcucha Tuska: {tuskScore}')
    # print(f'Był mylony z: {tuskPomylki}')
    #
    # print("Test tekstu Szydło")
    # szydloPredictions, szydloScore, szydloPomylki = makeLabels(szydloTest, 2, probabilities, labels)
    # print(f'Wynik łańcucha Szydło: {szydloScore}')
    # print(f'Był mylony z: {szydloPomylki}')
    #
    # print("Test tekstu Kopacz")
    # kopaczPredictions, kopaczScore, kopaczPomylki = makeLabels(kopaczTest, 3, probabilities, labels)
    # print(f'Wynik łańcucha Kopacz: {kopaczScore}')
    # print(f'Był mylony z: {kopaczPomylki}')