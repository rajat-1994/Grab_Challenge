import cv2
import numpy as np
from pathlib import Path
from PIL import Image
# from keras.models import model_from_json
import glob,os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def plot_batch_img(path, data, dic, batch=1):
    '''Plot batch of images of `path`'''
    batch = 16 * batch
    ims = [
        open_image(path / data.filename[i]) for i in range(batch - 16, batch)
    ]
    print(ims[0].shape)
    label = [data.target[i] for i in range(batch - 16, batch)]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        display_img(ims[i], dic=dic, ax=ax, label=label[i])
    plt.tight_layout(pad=0.1)


def display_img(im,
                dic,
                prediction=None,
                figsize=None,
                ax=None,
                alpha=None,
                label=0,
                ohe=False):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    if ohe:
        ax.text(-2,
                -2,
                '%.20s (%.2f)' %
                (dic[np.argmax(prediction)], np.max(prediction)),
                color='w',
                backgroundcolor='k',
                alpha=0.8)
    else:
        ax.text(-2,
                -2,
                '%.25s...' % dic[label],
                color='k',
                backgroundcolor='w',
                alpha=0.8)
    ax.set_axis_off()
    return ax


def open_image(path):
    '''Return image of path `fn` '''
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(path):
        raise OSError('No such file or directory: {}'.format(path))
    elif os.path.isdir(path):
        raise OSError('Is a directory: {}'.format(path))
    else:
        try:
            im = cv2.imread(str(path), flags).astype(np.float32) / 255.
            if im is None:
                raise OSError('File not recognized by opencv: %d', path)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e
# #prints precision and recall of each class
# def print_precision_recall(cm,class_dic):
#     '''Prints precision and recall of the predictions'''
#     for i in range(len(class_dic.keys())):
#         temp=cm[:,i][i]
#         print('precision of',class_dic[i],'is',np.round(temp/sum(cm[:,i]),2),'  and  recall','is',
#               np.round(temp/sum(cm[i]),2))

# def confusionmatrix(prediction,df,numclass=196):
#     '''Returns confusion matrix of the predictions'''
#     y_pred = np.argmax(prediction, axis=1)
#     cm = confusion_matrix(df.target,y_pred)
#     return cm

# def plot_confusionmatrix(prediction,dataframe,class_dic,numclass=196):
#     '''Plot comfusion matrix'''
#     cm = confusionmatrix(prediction,dataframe,numclass=196)
#     df_cm = pd.DataFrame(cm, range(196),range(196))
#     df_cm.index=class_dic.keys()
#     df_cm.columns=class_dic.keys()
#     plt.figure(figsize = (30,30))
#     sn.set(font_scale=1.4)#for label size
#     sn.heatmap(df_cm,cbar=False, annot=True,annot_kws={"size": 12})# font size
#     plt.show()
            
