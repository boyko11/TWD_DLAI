import numpy as np
from td_utils import match_target_amplitude, graph_spectrogram
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

class ModelService:

    def __init__(self):

        self.Ty = 1375

    def get_random_time_segment(self, segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """

        segment_start = np.random.randint(low=0,
                                          high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
        segment_end = segment_start + segment_ms - 1

        return (segment_start, segment_end)

    def is_overlapping(self, segment_time, previous_segments):
        """
        Checks if the time of a segment overlaps with the times of existing segments.

        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """

        segment_start, segment_end = segment_time

        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False

        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in previous_segments:
            if previous_start <= segment_start <= previous_end or segment_start <= previous_start <= segment_end:
                overlap = True

        return overlap

    def insert_audio_clip(self, background, audio_clip, previous_segments):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the
        audio segment does not overlap with existing segments.

        Arguments:
        background -- a 10 second background audio recording.
        audio_clip -- the audio clip to be inserted/overlaid.
        previous_segments -- times where audio segments have already been placed

        Returns:
        new_background -- the updated background audio
        """

        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)

        ### START CODE HERE ###
        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
        # the new audio clip. (≈ 1 line)
        segment_time = self.get_random_time_segment(segment_ms)

        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
        # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
        while self.is_overlapping(segment_time, previous_segments):
            segment_time = self.get_random_time_segment(segment_ms)

        # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)

        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position=segment_time[0])

        return new_background, segment_time

    def insert_ones(self, y, segment_end_ms):
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 following labels should be ones.


        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_end_ms -- the end time of the segment in ms

        Returns:
        y -- updated labels
        """

        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * self.Ty / 10000.0)

        # Add 1 to the correct index in the background label (y)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < self.Ty:
                y[0, i] = 1

        return y

    def create_training_example(self, background, activates, negatives):
        """
        Creates a training example with a given background, activates, and negatives.

        Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"

        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """

        # Set the random seed
        np.random.seed(18)

        # Make background quieter
        background = background - 20

        # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
        y = np.zeros(shape=(1, self.Ty))

        # Step 2: Initialize segment times as an empty list (≈ 1 line)
        previous_segments = []

        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(0, 5)
        random_indices = np.random.randint(len(activates), size=number_of_activates)
        random_activates = [activates[i] for i in random_indices]

        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for random_activate in random_activates:
            # Insert the audio clip on the background
            background, segment_time = self.insert_audio_clip(background, random_activate, previous_segments)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = self.insert_ones(y, segment_end)

        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(0, 3)
        random_indices = np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[i] for i in random_indices]

        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background
            background, _ = self.insert_audio_clip(background, random_negative, previous_segments)

        # Standardize the volume of the audio clip
        background = match_target_amplitude(background, -20.0)

        # Export new training example
        file_handle = background.export("train" + ".wav", format="wav")
        print("File (train.wav) was saved in your directory.")

        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        x = graph_spectrogram("train.wav")

        return x, y

    def model(self, input_shape):
        """
        Function creating the model's graph in Keras.

        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)

        Returns:
        model -- Keras model instance
        """

        X_input = Input(shape=input_shape)

        ### START CODE HERE ###

        # Step 1: CONV layer (≈4 lines)
        X = Conv1D(filters=196,kernel_size=15,strides=4)(X_input)  # CONV1D
        X = BatchNormalization()(X)  # Batch normalization
        X = Activation("relu")(X)  # ReLu activation
        X = Dropout(rate=0.8)(X)  # dropout (use 0.8)

        # Step 2: First GRU Layer (≈4 lines)
        X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
        X = Dropout(rate=0.8)(X)   # dropout (use 0.8)
        X = BatchNormalization()(X)  # Batch normalization

        # Step 3: Second GRU Layer (≈4 lines)
        X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
        X = Dropout(rate=0.8)(X)  # dropout (use 0.8)
        X = BatchNormalization()(X)  # Batch normalization
        X = Dropout(rate=0.8)(X)  # dropout (use 0.8)

        # Step 4: Time-distributed dense layer (see given code in instructions) (≈1 line)
        X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)  # time distributed  (sigmoid)

        ### END CODE HERE ###

        model = Model(inputs=X_input, outputs=X)

        return model