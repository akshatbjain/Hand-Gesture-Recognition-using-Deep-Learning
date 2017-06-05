from keras.utils.visualize_util import plot
from keras.models import model_from_json

# load json and create model
json_file = open('ModelA2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

plot(loaded_model, to_file='model.png')
