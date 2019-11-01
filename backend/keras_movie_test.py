import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt


np.random.seed(0)

number_of_features = 10000

(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)

features_train = tokenizer.sequences_to_matrix(data_train, mode = "binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode = "binary")

network = models.Sequential()

network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))

network.add(layers.Dense(units=16, activation="relu"))

network.add(layers.Dense(units=1, activation="sigmoid"))

network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = network.fit(features_train, target_train, epochs=15, verbose=0, batch_size=1000, validation_data=(features_test, target_test))


# print(history.history.keys())

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]
#
epoch_count = range(1, len(training_loss)+1)
#
# plt.plot(epoch_count, training_loss, "r--")
# plt.plot(epoch_count, test_loss, "b-")
# plt.legend(["Training Loss", "Test loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

training_accuracy = history.history["accuracy"]

test_accuracy = history.history["val_accuracy"]

plt.plot(epoch_count, training_accuracy, "r--")

plt.plot(epoch_count, test_accuracy, "b-")

plt.legend(["Training Accuracy", "Test Accuracy"])

plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")

plt.show()


predicted_target = network.predict(features_test)




print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in data_train)))
print('리뷰의 평균 길이 : {}'.format(sum(map(len, data_train))/len(data_train)))

plt.hist([len(s) for s in data_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
print(target_test[:50])
print(predicted_target[:50])
