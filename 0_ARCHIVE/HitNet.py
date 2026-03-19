from keras.layers import Input, Dense, Embedding, GRU, Concatenate, Softmax
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

#########################################################################
############# A CONSTRUIRE EN S'INSPIRANT DE L'ARTICLE###################
#########################################################################

# Define input tensors
num_frames = 12
num_features = 32

input_court = Input(shape=(num_frames, 8), name='input_court')  # Court coordinates
input_poses = Input(shape=(num_frames, 48), name='input_poses')  # Player poses
input_shuttle = Input(shape=(num_frames, 2), name='input_shuttle')  # Shuttlecock positions

# Embedding layer
embedding = Dense(num_features)(Concatenate()([input_court, input_poses, input_shuttle]))

# Recurrent unit
gru_1 = GRU(units=num_features, return_sequences=True)(embedding)
gru_2 = GRU(units=num_features)(gru_1)

# Fully connected layer
fc = Dense(num_features)(gru_2)

# Softmax layer for hit prediction
# hit_prediction = Dense(2, activation='softmax')(fc)
hit_prediction = Softmax()(fc)

# Define the model
model = Model(inputs=[input_court, input_poses, input_shuttle], outputs=hit_prediction)

# Compile the model
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Generate example data
# Replace this with your actual data loading and preprocessing steps
# Here, we create dummy data as an example
num_examples = 10000
input_court_array = np.random.rand(num_examples, num_frames, 8)
input_poses_array = np.random.rand(num_examples, num_frames, 48)
input_shuttle_array = np.random.rand(num_examples, num_frames, 2)
labels_array = np.random.randint(2, size=(num_examples,))

# Sliding window approach
stride = 1

input_court_examples = []
input_poses_examples = []
input_shuttle_examples = []
labels_examples = []

for i in range(0, len(input_court_array) - num_frames + 1, stride):
    input_court_example = input_court_array[i:i+num_frames]
    input_poses_example = input_poses_array[i:i+num_frames]
    input_shuttle_example = input_shuttle_array[i:i+num_frames]
    labels_example = labels_array[i+num_frames-1]

    input_court_examples.append(input_court_example)
    input_poses_examples.append(input_poses_example)
    input_shuttle_examples.append(input_shuttle_example)
    labels_examples.append(labels_example)

input_court_examples = np.array(input_court_examples)
input_poses_examples = np.array(input_poses_examples)
input_shuttle_examples = np.array(input_shuttle_examples)
labels_examples = np.array(labels_examples)

# Split the data into training and validation sets
input_court_train, input_court_val, input_poses_train, input_poses_val, input_shuttle_train, input_shuttle_val, labels_train, labels_val = train_test_split(input_court_examples, input_poses_examples, input_shuttle_examples, labels_examples, test_size=0.2, random_state=42)


# Train the model
model.fit(x=[input_court_train, input_poses_train, input_shuttle_train], y=labels_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x=[input_court_val, input_poses_val, input_shuttle_val], y=labels_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)