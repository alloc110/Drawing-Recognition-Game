from dataset import Mydataset
import matplotlib.pyplot as plt
data = Mydataset('dataset')

categories = ['apple',
                      'bird',
                      'bread',
                      'cake',
                      'car',
                      'elephant',
                      'fish',
                      'hat',
                      'lion',
                      'monkey',
                      'rabbit'
                      ]

# image, label = data.__getitem__(300000)
# plt.imshow(image)
# plt.show()
# print(categories[label])


