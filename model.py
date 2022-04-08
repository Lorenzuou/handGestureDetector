import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


le = preprocessing.LabelEncoder()

data = pd.read_csv("HandData.csv")



target = '42'
data[target] = le.fit_transform(data[target])


y = data[target] # class colum
X = data.drop(columns=['Unnamed: 0',target])
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)


scaler = MinMaxScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



from pickle import dump
dump(scaler, open('scaler.pkl', 'wb'))




def neural_model(X_train_scaled, y_train) :

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='sparse_categorical_crossentropy'
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=1000,
        batch_size=100,
        validation_data=(X_test_scaled, y_test),

    )

    val_loss, val_acc = model.evaluate(X_test_scaled, y_test, batch_size=128)

    print(val_loss)
    print(val_acc)

    return model



model = neural_model(X_train_scaled, y_train)

model.save("model.hdf5", include_optimizer=False)

