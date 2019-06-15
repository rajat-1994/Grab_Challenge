import cv2
import numpy as np
from pathlib import Path
from PIL import Image
# from keras.models import model_from_json
import glob,os
import tensorflow as tf
import tensorflow.keras as K

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def plot_batch_img(path,data,dic,batch=1):
    batch = 16*batch
    ims = [open_image(path/data.filename[i]) for i in range(batch-16,batch)]
    print(ims[0].shape)
    label=[data.target[i] for i in range(batch-16,batch)]
    fig, axes = plt.subplots(4, 4,figsize=(10,10))
    for i,ax in enumerate(axes.flat): display_img(ims[i],dic=dic, ax=ax,label=label[i])
    plt.tight_layout(pad=0.1)

def display_img(im,dic,prediction=None,figsize=None, ax=None, alpha=None,label=0,ohe=False):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    if ohe:
        ax.text(-2,-2, '%.20s (%.2f)' % (dic[np.argmax(prediction)],np.max(prediction)), 
                                                        color='w', backgroundcolor='k', alpha=0.8)
    else : ax.text(-2,-2, '%.25s...' % dic[label], color='k', backgroundcolor='w', alpha=0.8)
    ax.set_axis_off()
    return ax


#ploting confusing matrix     
def plot_cm(cm,l_to_n):
    df_cm = pd.DataFrame(cm, range(7),
                                range(7))
    df_cm.index=l_to_n.keys()
    df_cm.columns=l_to_n.keys()
    plt.figure(figsize = (10,10))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm,cbar=False, annot=True,annot_kws={"size": 12})# font size
    plt.show()

#prints precision and recall of each class
def get_pr(cm,n_to_l):
    for i in range(len(n_to_l.keys())):
        temp=cm[:,i][i]
        print('precision of',n_to_l[i],'is',np.round(temp/sum(cm[:,i]),2),'  and  recall','is',
              np.round(temp/sum(cm[i]),2))

def open_image(fn):

    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError('File not recognized by opencv: %d',fn)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e
            
