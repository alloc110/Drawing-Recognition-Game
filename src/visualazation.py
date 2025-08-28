import matplotlib.pyplot as plt
import numpy as np

data_train = np.load('dataset/train_label.npy')
data_test = np.load('dataset/test_label.npy')
map = [0] * 11
for i in data_test:
    map[i] += 1

for i in data_train:
    map[i] += 1

len = len(data_train) + len(data_test)
categories =  ['apple',
                      'cake',
                      'rabbit',
                      'fish',
                      'bread',
                      'monkey',
                      'hat',
                      'elephant',
                      'bird',
                      'lion',
                      'car'
                      ]
plt.pie(map, labels = categories)
plt.title('Class Distribution')
plt.axis('equal')
plt.legend()
plt.show()
plt.savefig("pie_chart_train.png")
for idx, i in enumerate(map):
    print('{0}: {1:.3f}%'.format(categories[idx] , i * 100 / len))
