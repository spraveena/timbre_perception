#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the model architecture and weights
input_size = 5
hidden_size = 64
output_size = 5
trial_length = 1000

# Rebuild the model structure
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=hidden_size, activation='tanh', return_sequences=True, input_shape=(trial_length, input_size)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size, activation='linear'))
])

# Load model weights
model.load_weights('best_model.h5')
print("Model loaded successfully.")

# Generate a single example stimulus
def generate_example_stimulus(trial_length, stimulus_length, num_classes=5):
    stimulus_class = np.random.choice(num_classes)
    example_input = np.zeros((trial_length, num_classes))
    example_input[:stimulus_length, :] = np.eye(num_classes)[stimulus_class]
    return example_input, stimulus_class

stimulus_length = 100

example_input, stimulus_class = generate_example_stimulus(trial_length, stimulus_length)
example_input_batch = np.expand_dims(example_input, axis=0)  # Add batch dimension

# Get model predictions and intermediate RNN layer output
intermediate_model = tf.keras.Model(inputs=model.input,
                                    outputs=[model.layers[0].output, model.output])

rnn_activities, output_activities = intermediate_model.predict(example_input_batch)

# Plot the input stimulus
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title(f"Input Stimulus (Stimulus Class: {stimulus_class})")
plt.imshow(example_input.T, aspect='auto', cmap='hot')
plt.ylabel("Stimuli")
plt.colorbar()

# Plot all RNN unit activities
plt.subplot(3, 1, 2)
plt.title("RNN Unit Activities")
plt.imshow(rnn_activities[0].T, aspect='auto', cmap='viridis')
plt.ylabel("RNN Units (64)")
plt.colorbar()

# Plot the output activity (output layer)
plt.subplot(3, 1, 3)
plt.title("Output Activities")
plt.imshow(output_activities[0].T, aspect='auto', cmap='cool')
plt.xlabel("Time Steps")
plt.ylabel("Output Units (5)")
plt.colorbar()

plt.tight_layout()
plt.show()