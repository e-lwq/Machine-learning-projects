from PIL import Image
import numpy as np

resize_SIZE = (50,50)

X_arr = []
Y_arr = []

filepath = '/Users/eeeeeee/Desktop/spyder_folder/cats_vs_dogs/img/'

for i in range(12500):
    for animal in ['Cat/','Dog/']:
        if i==666 and animal=='Cat/' or i==11702 and animal=='Dog/':
            continue
        img = Image.open(filepath + animal + str(i) + '.jpg')
        img = img.convert('RGB')
        img = img.quantize(colors=256, method=Image.FASTOCTREE)

        w,h = img.size

        if w>h:
            diff = (w-h)//2
            img = img.crop((diff,0,w-diff,h))
        elif h>w:
            diff = (h-w)//2
            img = img.crop((0,diff,w,h-diff))

        #img = img.resize((250,250))
        #img = img.crop((25,25,225,225))

        img = img.resize(resize_SIZE)

        #pixels = np.asarray(img, dtype = 'int32')
        pixels=[]
        for j in range(img.size[0]):
            for k in range(img.size[1]):
              try:
                p = img.getpixel((j,k))
                pixels.append(p)
              except:
                print(i,animal)

        X_arr.append(pixels)

        if animal=='Cat/':
            Y_arr.append(0)
        else:
            Y_arr.append(1)
    print(i, end=' ')
    if i%100==0:
        print()
    
for i in range(len(X_arr)):
  X_arr[i].append(Y_arr[i])
  

import csv

col_names = []
for i in range(2500):
  col_names.append(str(i))
col_names.append('target')
#print(col_names)

with open('csv_of_1000img.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(col_names)
    writer.writerows(X_arr)