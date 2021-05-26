import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#import data
df = pd.read_excel('./dataset/knn-ppdb.xlsx', engine='openpyxl')

#tampilkan data
df.head()

#cek data kosong
df.isna().sum()

#bagi dataset jadi atribut dan label
y = df['status'].values #label
X = df.iloc[:,:-1].values #atribut

#normalisasi menggunakan minmax
scaler = MinMaxScaler()
scaler.fit(X)
scaler.transform(X)

#fungsi untuk KFold Cross Validation dg jumlah k=5
# Kfoldcv(jumlah kfold, jumlah k pd knn)
from sklearn.model_selection import KFold

ss = KFold(n_splits=5)
for train_index, test_index in ss.split(X):
    #print("%s %s" % (train_index, test_index))
    #bagi data training dan testing
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # build model
    model = KNeighborsClassifier(n_neighbors=5) #panggil fungsi KNN dengan jumlah k sesuai dengan parameter k
    model.fit(X_train,y_train) # training model/classifier
    y_pred = model.predict(X_test) # melakukan prediksi
    #cek hasil confusion matrix
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    print("\n")

#testing menggunakan dataset random
tes = [[1,3,0.25,78.89,1,0,0,0.0]]
print(model.predict(tes))

#digunakan untuk API
import pickle
pickle.dump(model, open('model.pickle', 'wb'))
