import scipy.io
import pandas as pd

data = scipy.io.loadmat('./cars_annos.mat')
data = data['annotations'][0]

target =[data[i][-2][0][0] for i in range(data.shape[0])]
image_id =[data[i][0][0] for i in range(data.shape[0])]
is_train = [data[i][-1][0][0] for i in range(data.shape[0])]

df = pd.DataFrame()
df['filename'] = image_id
df['target'] = target
df['is_train'] = is_train

train_df = df[df.is_train==1]
test_df = df[df.is_train==0]

train_df.drop('is_train',axis=1,inplace=True)
test_df.drop('is_train',axis=1,inplace=True)

train_df = train_df.sample(frac=1,random_state=10)
test_df = test_df.sample(frac=1,random_state=10)

train_df.to_csv('./train.csv',index=False)
test_df.to_csv('./test.csv',index=False)