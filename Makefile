SRC='src'

train:
	python3 ${SRC}/gan_mnist_train.py

test:
	python3 ${SRC}/gan_mnist_test.py
