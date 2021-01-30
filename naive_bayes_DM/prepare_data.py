import pandas as pd
import numpy as np
import ast
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

data = pd.read_csv("emails.csv")
file = open("vocabulary.txt", "r")
contents = file.read()
vocabulary = ast.literal_eval(contents)

X = np.zeros((data.shape[0], len(vocabulary)))
y = np.zeros((data.shape[0]))

for i in range(data.shape[0]):
    # wszystkie wyrazy z danego email'a
    email = data.iloc[i, :][0].split()

    for email_word in email:
        if email_word.lower() in vocabulary:
            X[i, vocabulary[email_word]] += 1

    y[i] = data.iloc[i, :][1]


# print(f'Example => {X[1240, :]}')
# Save stored numpy arrays

np.save("X.npy", X)
np.save("y.npy", y)