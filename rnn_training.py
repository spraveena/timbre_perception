import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate one-hot encoded stimuli
def generate_one_hot_stimuli(num_stimuli):
    return np.eye(num_stimuli)

# Generate dataset: Trials of length 1000 time steps
def generate_dataset(num_trials, trial_length, stimulus_length, num_classes=5):
    stimuli = generate_one_hot_stimuli(num_classes)
    data = []  # trials
    labels = []  # extended stimulus classes (0-4) with zeros for no stimulus
    
    for _ in range(num_trials):
        stimulus_class = np.random.choice(num_classes)
        trial = np.zeros((trial_length, num_classes))
        # Place the stimulus in the first stimulus_length milliseconds
        trial[:stimulus_length, :] = stimuli[stimulus_class]
        # Labels are zeros for the first 900ms and the stimulus class for the last 100ms
        label = np.zeros((trial_length, num_classes))
        label[900:1000, :] = stimuli[stimulus_class]  # Labels are the stimulus for the last 100ms
        data.append(trial)
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Hyperparameters
input_size = 5
hidden_size = 64
output_size = 5
num_epochs = 10
batch_size = 20
learning_rate = 0.001
trial_length = 1000
stimulus_length = 100
num_trials = 5000  # 1000 trials for each of 5 stimuli

# Generate dataset
data, labels = generate_dataset(num_trials, trial_length, stimulus_length)


# Build the model
model = Sequential()
# Add a SimpleRNN layer with tanh activation (which is the default activation function)
model.add(SimpleRNN(units=hidden_size, activation='tanh', return_sequences=True, input_shape=(trial_length, input_size)))
# Add a TimeDistributed Dense layer with linear activation (which is the default activation function)
model.add(TimeDistributed(Dense(output_size, activation='linear')))

# Define a custom loss function to ensure the output is zero from 0 to 900ms and match the target from 900ms to 1000ms
def custom_mse_loss(y_true, y_pred):
    mse_loss_true_pred = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.5 * mse_loss_true_pred


# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_mse_loss)

# Checkpoint to save the best model
checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True, mode='min', save_weights_only=True)

# Train the model
history = model.fit(data, labels, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint])

# Load the best model weights
model.load_weights('best_model.h5')
print("Training complete.")