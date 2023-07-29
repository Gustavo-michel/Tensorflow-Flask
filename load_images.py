from tensorflow.keras.datasets import fashion_mnist
from imageio import imwrite

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imwrite(f"uploads/{i}.png",X_test[i])