import numpy as np

from keras.models import load_model

trained_network = load_model('trained_network.h5')

print(f"model input shape: {trained_network.input_shape}")
print(f"model.output_shape: {trained_network.output_shape}")

# create an array for input
# start with all zeros
input_data = np.zeros((1, 10))

for i in range(3):
    print(f"------------- step {i} -------------")

    # make a prediction based on input_data
    predicted_output = trained_network.predict(input_data)
    predicted_best_action = np.argmax(predicted_output)

    print(f"input_data            : {input_data}")
    print(f"predicted_output      : {predicted_output}")
    print(f"predicted best action : {predicted_best_action}")

    # record the action taken
    input_data = np.roll(input_data, shift=2)
    input_data[0, 0] = np.random.randn()  # this should be observed data
    input_data[0, 1] = predicted_best_action

