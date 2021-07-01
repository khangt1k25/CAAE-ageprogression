import os
import os
from PIL import Image
import numpy as np
import pickle

origin_dir = "./UTKFace"
des_dir = "./compress"
imgFiles = [file for file in os.listdir(origin_dir)]

def encodeAge(n):
    if n<=5:
        return 0
    elif n<=10:
        return 1
    elif n<=15:
        return 2
    elif n<=20:
        return 3
    elif n<=30:
        return 4
    elif n<=40:
        return 5
    elif n<=50:
        return 6
    elif n<=60:
        return 7
    elif n<=70:
        return 8
    else:
        return 9



def compress():
    all = []
    for file in imgFiles:
        
        cur = {}
        lst = file.split("_")
        age = int(lst[0])
        gender = int(lst[1])
        label = encodeAge(age)
        if (gender==1):
            label += 10
        
        cur['label'] = label

        
        image = Image.open(os.path.join(origin_dir, file))
        cur['img'] = np.array(image)
        all.append(cur)
        image.close()

    with open('./compress/all_utk.pkl', 'wb') as f:
        pickle.dump(all, f)
if not os.path.exists(des_dir):
    os.mkdir(des_dir)

compress()