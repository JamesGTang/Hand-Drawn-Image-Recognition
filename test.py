import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create a numerical classes list
classes = open('class.txt').read().splitlines()

# load image data
data = np.load("train_images.npy",encoding='latin1')
# load label 
label = pd.read_csv('train_labels.csv',sep=',')

# replace classifer for each image to numerical
for index, row in label.iterrows():
	class_at_i = label.iloc[index,1]
	# print(class_at_i)
	label.at[index,'Category'] = classes.index(class_at_i)

print(label.head())
# img1 =(data[0][1]).reshape(100,100)
# imgplot = plt.imshow(img1)

# # search for where id = 2 and find the value of id=2
# plt.title(label.loc[2].iloc[1])
# plt.show()