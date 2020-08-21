from flask import Flask, render_template, request
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def classify(imageFath):
	np.set_printoptions(suppress=True)

	model = tensorflow.keras.models.load_model('./model/keras_model.h5')

	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

	image = Image.open(imageFath)

	size = (224, 224)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	image_array = np.asarray(image)

	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	data[0] = normalized_image_array

	prediction = model.predict(data)
	
	if prediction[0][0] > 0.7:
		return "김다미"
	elif prediction[0][1] > 0.7:
		return "안지영"
	else:
		return "판독 불가"

app = Flask(__name__)

@app.route('/')
def hello():
	return '<h1>Hello world!</h1>'

@app.route('/user/<name>')
def user(name):
	return render_template('user.html', name=name)

@app.route('/classify')
def classifyKA():
	return render_template('cl.html')

@app.route('/classify/who', methods = ['GET', 'POST'])
def checkKA():
	if request.method == 'POST':
		value = request.files['imageFile']
		who = classify(value)
		return render_template('cl.html', who=who)

if __name__ == '__main__':
	app.run(debug=True)