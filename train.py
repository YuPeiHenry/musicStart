import numpy as np
import tensorflow as tf
import os
import msgpack
import glob
from tqdm import tqdm
import midi_manipulation

lowest_note = midi_manipulation.lowerBound  # the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound  # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

step_size = 8
num_timesteps = 16
max_size = 80
shuffle_buffer_size = 99999
batch_size = 16
tf.logging.set_verbosity(tf.logging.INFO)

def parse_fn(midi, label):
	offset = np.random.randint(max_size - num_timesteps + 1)
	return (midi[offset: offset + num_timesteps], label)

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] >= max_size:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

def chop_real(songs):
	new_songs = []
	for song in songs:
		song = np.array(song)
		for i in range(int(np.shape(song)[0] / max_size) * 2 - 1) :
			offset = i * int(max_size / 2)
			piece = song[offset: offset + max_size]
			piece = np.reshape(piece, [max_size, note_range * 2])
			new_songs.append(piece)
	labels = [[1] for song in new_songs]
	return (new_songs, labels)
	
real_song = []
fake_song = []
real_label = []
fake_label = []
def get_real_and_fake():
	global real_song, fake_song, real_label, fake_label
	if (len(fake_song) == 0):
		(fake_song, _) = get_noise(np.shape(real_song)[0])
		fake_label = [[0] for song in fake_song]
	midis = np.concatenate((real_song, fake_song), axis=0)
	labels = np.concatenate((real_label, fake_label), axis=0)
	
	dataset = tf.data.Dataset.from_tensor_slices((midis, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	return dataset

def get_noise(size):
	return (np.round(np.random.rand(size, max_size, note_range * 2)), [[1] for i in range(size)])

def gen_data():
	(midis, labels) = get_noise(np.shape(real_song)[0])
	
	dataset = tf.data.Dataset.from_tensor_slices((midis, labels))
	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		map_func=parse_fn, batch_size=batch_size))
	return dataset
	
def create_new_conv_layer(input_data, weights, bias):
	out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
	out_layer += bias

	return out_layer

def create_pool_layer(input_data, pool_strideX, pool_strideY):
	ksize = [1, pool_strideX, pool_strideY, 1]
	strides = [1, pool_strideX, pool_strideY, 1]
	out_layer = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding='SAME')
	
	return out_layer
	
def discriminator(features, labels, mode):
	"""Model function for CNN."""
	with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
		# Input Layer
		# Reshape X to 4-D tensor: [batch_size, length, range, channels]
		input_layer = tf.reshape(features, [-1, num_timesteps, note_range * 2, 1])

		########################################
		
		convA1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convA1",
			reuse=tf.AUTO_REUSE)
		convA2 = tf.layers.conv2d(
			inputs=convA1,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convA2",
			reuse=tf.AUTO_REUSE)
		convA3 = tf.layers.conv2d(
			inputs=convA2,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convA3",
			reuse=tf.AUTO_REUSE)
		
		convB1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convB1",
			reuse=tf.AUTO_REUSE)
		convB2 = tf.layers.conv2d(
			inputs=convB1,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convB2",
			reuse=tf.AUTO_REUSE)
		
		convC1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Gen_convC1",
			reuse=tf.AUTO_REUSE)
		
		combined = tf.concat([convA3, convB2, convC1], 3)
		pool = create_pool_layer(combined, 4, 12)
		pool_flat = tf.reshape(pool, [-1, int(num_timesteps * note_range * 2 * 12 / 4 / 12)])
		########################################################################

		denseA = tf.layers.dense(inputs=pool_flat, units=64, activation=tf.nn.sigmoid, name="Gen_denseA", reuse=tf.AUTO_REUSE)
		recent = tf.reshape(tf.slice(input_layer, [0, num_timesteps - 1, 0, 0], [-1, 1, note_range * 2, 1]), [-1, note_range * 2])
		recent_combined = tf.concat([denseA, recent], 1)
		denseB = tf.layers.dense(inputs=recent_combined, units=note_range * 2, activation=tf.nn.sigmoid, name="Gen_denseB", reuse=tf.AUTO_REUSE)
		########################################################################

		convA1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA1",
			reuse=tf.AUTO_REUSE)
		convA2 = tf.layers.conv2d(
			inputs=convA1,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA2",
			reuse=tf.AUTO_REUSE)
		convA3 = tf.layers.conv2d(
			inputs=convA2,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA3",
			reuse=tf.AUTO_REUSE)
		
		convB1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convB1",
			reuse=tf.AUTO_REUSE)
		convB2 = tf.layers.conv2d(
			inputs=convB1,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convB2",
			reuse=tf.AUTO_REUSE)
		
		convC1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convC1",
			reuse=tf.AUTO_REUSE)
		
		combined = tf.concat([convA3, convB2, convC1], 3)
		pool = create_pool_layer(combined, 4, 12)
		pool_flat = tf.reshape(pool, [-1, int(num_timesteps * note_range * 2 * 12 / 4 / 12)])
		########################################################################

		denseA = tf.layers.dense(inputs=pool_flat, units=64, activation=tf.nn.sigmoid, name="Dis_denseA", reuse=tf.AUTO_REUSE)
		denseB = tf.layers.dense(inputs=denseA, units=1, activation=None, name="Dis_denseB", reuse=tf.AUTO_REUSE)

		predictions = {
			"probabilities": tf.nn.sigmoid(denseB, name="sigmoid_tensor"),
		}
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=denseB)

		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0025)
			#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step(),
				var_list=[tf.get_variable("Dis_convA1/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convA2/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convA3/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convB1/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convB2/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convC1/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_denseA/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_denseB/kernel", dtype="float64_ref"),
					tf.get_variable("Dis_convA1/bias", dtype="float64_ref"),
					tf.get_variable("Dis_convA2/bias", dtype="float64_ref"),
					tf.get_variable("Dis_convA3/bias", dtype="float64_ref"),
					tf.get_variable("Dis_convB1/bias", dtype="float64_ref"),
					tf.get_variable("Dis_convB2/bias", dtype="float64_ref"),
					tf.get_variable("Dis_convC1/bias", dtype="float64_ref"),
					tf.get_variable("Dis_denseA/bias", dtype="float64_ref"),
					tf.get_variable("Dis_denseB/bias", dtype="float64_ref")])
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=tf.round(predictions["probabilities"]))}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def generator(features, labels, mode):
	"""Model function for CNN."""
	with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
		# Input Layer
		# Reshape X to 4-D tensor: [batch_size, length, range, channels]
		input_layer = tf.reshape(features, [-1, num_timesteps, note_range * 2, 1])

		next_input = input_layer
		pieces = [None] * num_timesteps
		for i in range(-1, num_timesteps):
			GconvA1 = tf.layers.conv2d(
				inputs=next_input,
				filters=2,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convA1",
				reuse=tf.AUTO_REUSE)
			GconvA2 = tf.layers.conv2d(
				inputs=GconvA1,
				filters=2,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convA2",
				reuse=tf.AUTO_REUSE)
			GconvA3 = tf.layers.conv2d(
				inputs=GconvA2,
				filters=4,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convA3",
				reuse=tf.AUTO_REUSE)
			
			GconvB1 = tf.layers.conv2d(
				inputs=next_input,
				filters=2,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convB1",
				reuse=tf.AUTO_REUSE)
			GconvB2 = tf.layers.conv2d(
				inputs=GconvA2,
				filters=4,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convB2",
				reuse=tf.AUTO_REUSE)
			
			GconvC1 = tf.layers.conv2d(
				inputs=next_input,
				filters=4,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu,
				name="Gen_convC1",
				reuse=tf.AUTO_REUSE)
			
			Gcombined = tf.concat([GconvA3, GconvB2, GconvC1], 3)
			Gpool = create_pool_layer(Gcombined, 4, 12)
			Gpool_flat = tf.reshape(Gpool, [-1, int(num_timesteps * note_range * 2 * 12 / 4 / 12)])
			########################################################################

			GdenseA = tf.layers.dense(inputs=Gpool_flat, units=64, activation=tf.nn.sigmoid, name="Gen_denseA", reuse=tf.AUTO_REUSE)
			recent = tf.reshape(tf.slice(next_input, [0, num_timesteps - 1, 0, 0], [-1, 1, note_range * 2, 1]), [-1, note_range * 2])
			recent_combined = tf.concat([GdenseA, recent], 1)
			GdenseB = tf.layers.dense(inputs=recent_combined, units=note_range * 2, activation=tf.nn.sigmoid, name="Gen_denseB", reuse=tf.AUTO_REUSE)
			
			residue = tf.slice(next_input, [0, 1, 0, 0], [-1, num_timesteps - 1, note_range * 2, 1])
			new_notes = tf.expand_dims(GdenseB, 1)
			new_notes = tf.expand_dims(new_notes, 3)
			if i >= 0:
				pieces[i] = new_notes
			next_input = tf.concat([residue, new_notes], 1)

		###############################################################
		generated_output = tf.concat(pieces, 1)
		convA1 = tf.layers.conv2d(
			inputs=generated_output,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA1",
			reuse=tf.AUTO_REUSE)
		convA2 = tf.layers.conv2d(
			inputs=convA1,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA2",
			reuse=tf.AUTO_REUSE)
		convA3 = tf.layers.conv2d(
			inputs=convA2,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convA3",
			reuse=tf.AUTO_REUSE)
		
		convB1 = tf.layers.conv2d(
			inputs=generated_output,
			filters=2,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convB1",
			reuse=tf.AUTO_REUSE)
		convB2 = tf.layers.conv2d(
			inputs=convB1,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convB2",
			reuse=tf.AUTO_REUSE)
		
		convC1 = tf.layers.conv2d(
			inputs=generated_output,
			filters=4,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu,
			name="Dis_convC1",
			reuse=tf.AUTO_REUSE)
		
		combined = tf.concat([convA3, convB2, convC1], 3)
		pool = create_pool_layer(combined, 4, 12)
		pool_flat = tf.reshape(pool, [-1, int(num_timesteps * note_range * 2 * 12 / 4 / 12)])
		########################################################################

		denseA = tf.layers.dense(inputs=pool_flat, units=64, activation=tf.nn.sigmoid, name="Dis_denseA", reuse=tf.AUTO_REUSE)
		denseB = tf.layers.dense(inputs=denseA, units=1, activation=None, name="Dis_denseB", reuse=tf.AUTO_REUSE)

		predictions = {
			"output": tf.round(generated_output),
			"probabilities": tf.nn.sigmoid(denseB, name="sigmoid_tensor")
		}
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=denseB)

		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
			#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step(),
					var_list=[tf.get_variable("Gen_convA1/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convA2/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convA3/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convB1/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convB2/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convC1/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_denseA/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_denseB/kernel", dtype="float64_ref"),
					tf.get_variable("Gen_convA1/bias", dtype="float64_ref"),
					tf.get_variable("Gen_convA2/bias", dtype="float64_ref"),
					tf.get_variable("Gen_convA3/bias", dtype="float64_ref"),
					tf.get_variable("Gen_convB1/bias", dtype="float64_ref"),
					tf.get_variable("Gen_convB2/bias", dtype="float64_ref"),
					tf.get_variable("Gen_convC1/bias", dtype="float64_ref"),
					tf.get_variable("Gen_denseA/bias", dtype="float64_ref"),
					tf.get_variable("Gen_denseB/bias", dtype="float64_ref")])
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=tf.round(predictions["probabilities"]))}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
			
def main(unused_argv):
	global real_song, fake_song, real_label, fake_label
	(real_song, real_label) = chop_real(get_songs('Pop_Music_Midi'))  # These songs have already been converted from midi to msgpack

	# Create the Estimator
	classifier = tf.estimator.Estimator(
		model_fn=discriminator, model_dir="model/train")
	gen = tf.estimator.Estimator(
		model_fn=generator, model_dir="model/train")
	
	current_step = 160
	classifer_acc = 0
	while True:
		while classifer_acc <= 0.5:
			current_step += step_size
			train_spec = tf.estimator.TrainSpec(
				input_fn=get_real_and_fake,
				hooks=None,
				max_steps=current_step
			)
			val_spec = tf.estimator.EvalSpec(
				input_fn=get_real_and_fake,
				hooks=None,
				throttle_secs=3000,
				start_delay_secs=3000
			)
			tf.estimator.train_and_evaluate(
				classifier,
				train_spec,
				val_spec
			)
			classifer_acc = classifier.evaluate(input_fn=get_real_and_fake)["accuracy"]
		
		while classifer_acc > 0.5:
			current_step += step_size * 9
			Gtrain_spec = tf.estimator.TrainSpec(
				input_fn=gen_data,
				hooks=None,
				max_steps=current_step
			)
			Gval_spec = tf.estimator.EvalSpec(
				input_fn=gen_data,
				hooks=None,
				throttle_secs=3000,
				start_delay_secs=3000
			)
			tf.estimator.train_and_evaluate(
				gen,
				Gtrain_spec,
				Gval_spec
			)
			new_outputs = gen.predict(input_fn=gen_data)
			fake_song = []
			randIndex = np.random.randint(len(real_song))
			count = 0
			for new_output in new_outputs:
				if count == randIndex:
					sequence = np.reshape(new_output["output"], (num_timesteps, note_range * 2))
					midi_manipulation.noteStateMatrixToMidi(sequence, "out/generated_sequence_{}".format(current_step))
				count += 1
				fake_song.append(np.reshape(np.tile(new_output["output"], (5, 1, 1)), (max_size, note_range * 2)))
			classifer_acc = classifier.evaluate(input_fn=get_real_and_fake)["accuracy"]

if __name__ == "__main__":
  tf.app.run()
