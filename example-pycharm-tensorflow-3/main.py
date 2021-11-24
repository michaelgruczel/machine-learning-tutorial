import tensorflow as tf
import numpy as np
import pathlib as pl
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

if __name__ == '__main__':

  # we will let a very simple neural network learn fibonacci numbers
  input_values =      np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0], dtype=float)
  prediction_values = np.array([ 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 55.0], dtype=float)

  # on layer should do it
  learning_layer = Dense(units=1, input_shape=[1])
  model = Sequential([learning_layer])
  model.compile(optimizer='sgd', loss='mean_squared_error')
  model.fit(input_values, prediction_values, epochs=500)

  # let's see what it predicts for 8.0, the correct value would be 21
  print('\nPrediction for 8 (correct value is 21):', model.predict([8.0]))

  # export the learned weights, so that we can later use them in tensor serving and TF Lite
  print("learned weights {}".format(learning_layer.get_weights()))
  export_dir = 'learned_model/1'
  tf.saved_model.save(model, export_dir)
  converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
  tflite_model = converter.convert()
  tflite_model_file = pl.Path('model.tflite')
  tflite_model_file.write_bytes(tflite_model)

  # test whether loading model would work
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  to_predict = np.array([[8.0]], dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], to_predict)
  interpreter.invoke()
  tflite_results = interpreter.get_tensor(output_details[0]['index'])
  print('\nPrediction for 8 after importing weights from export (correct value is 21):', tflite_results)
