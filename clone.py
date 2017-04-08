
import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
import sklearn



# Global Constants
DATA_CHECK_ONLY = False
DEBUGGING = True
USING_AWS = False
BATCH_SIZE = 128
MIN_STEERING_ANGLE = 0.e
STEERING_ANGLE_ADJUSTMENT = 0.3
DELETE_RATE = 0.7
TARGET_SIZE = (64,64)



def displayCV2(image):
    '''
    Helper function to display CV2 images for troubleshooting only
    Takes an image and displays it
    '''
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_plots(history_object, using_AWS=True):
	'''
	Print the model MSE per epoch
	But only if local, since I can't get it to work on AWS
	'''
	print('Printing history_object details...')
	print(history_object.history.keys())
	print('Loss...')
	print(history_object.history['loss'])
	#print('Val_mean_squared_error...')
	#print(history.object.history['val_mean_squared_error'])
	#print('Val_loss:', history_object.history['val_loss'])

	if using_AWS==False:
		plt.plot(history_object.history['mean_squared_error'])
		plt.plot(history_object.history['val_mean_squared_error'])
		plt.title('model mean squared error')
		plt.ylabel('mean squared error')
		plt.xlabel('epoch')
		plt.legend(['training set', 'validation set'], loc='upper right')
		plt.show()



def load_data(del_rate=DELETE_RATE, cut_value=MIN_STEERING_ANGLE):
	'''
	Load each line of the CSV file and build array of image paths and angles.
	Skip most of the low angle images, except take a few to preserve some data
	Adjust the angles of the left and right camera images.
	NOTE: I don't load images here, just the image paths.
	I also load a flip_flag for each image path so that I can flip that image
	later in the Generator().
	Return lists image paths, steering measurements, and flip flags for train & validation
	'''

	# line[0] is path to center image
	# line[1] is left image; add to steering angle
	# line[2] is right image; subtract from steering angle
	# line[3] is steering angle

	lines = []  # Contains all the lines in the csv file

	# Note: This code for Windows... if uploading to AWS or Unix, change \\ to /
	#with open('./data-exp-22/driving_log.csv') as csvfile:
	#with open('./data/driving_log.csv') as csvfile:
	#with open('..\\CarND Simulator\\data\\driving_log.csv') as csvfile:
	#with open('..\\CarND Simulator\\sample_data\\data\\driving_log.csv') as csvfile:
	with open('./sample_data/driving_log.csv') as csvfile:
		row_count=0
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	
	del lines[0]	# Top row is a header row		

	image_paths = []
	measurements = []
	flip_flags = []
	skip_count = 0

	for line in lines:	
		if abs(float(line[3])) < cut_value:		
			if np.random.random() < del_rate:		
				skip_count += 1
				continue			# Skip most of the lines with small angles, but take some

		for i in range(3):  	# go through 3 camera angles in the row
			source_path = line[i]
			#filename = source_path.split('\\')[-1]   # File names in Win format so use \\
			filename = source_path.split('/')[-1] 
			#current_path = './sample_data/data/IMG/' + filename
			#current_path = './data-exp-22/IMG/' + filename			
			#current_path = './data/IMG/' + filename
			#current_path = '..\\CarND Simulator\\data\\IMG\\' + filename
			#current_path = '..\\CarND Simulator\\sample_data\\data\\IMG\\' + filename
			current_path = './sample_data/IMG/' + filename
			if not os.path.isfile(current_path):
				print('WARNING: IMAGE IS MISSING...', current_path)
				break
			if DEBUGGING:
				print('Current path:', current_path)
					
			# Load image path first time, with flip_flag=False
			image_paths.append(current_path)
			if (i==0): 		# center camera
				measurement = float(line[3]) 
			elif (i==1): 	# left camera
				measurement = float(line[3]) + STEERING_ANGLE_ADJUSTMENT
			elif (i==2): 	#r ight	camera
				measurement = float(line[3]) - STEERING_ANGLE_ADJUSTMENT
			measurements.append(measurement)
			flip_flags.append(False)

			# Load image path again, with flip_flag=True -- flip later in Generator()
			image_paths.append(current_path)
			measurements.append(measurement)
			flip_flags.append(True)

	#### END LOADING				

	data = np.column_stack((image_paths, measurements, flip_flags))
	data = sklearn.utils.shuffle(data)

	train_samples, validation_samples = train_test_split(data, test_size=0.2)

	if DEBUGGING:
		print('Total lines read in file:', len(lines))
		print('Total lines skipped:', skip_count)
		print('Total image paths loaded (including flipped):', len(image_paths))
		print('Total measurements loaded (included flipped):', len(measurements))
		print('Total flip_flags loaded:', len(flip_flags))
		print()
		print('Num train samples:', len(train_samples))
		print('Shape of train_samples:', train_samples.shape)
		print()									
		print('Num validation samples:', len(validation_samples))
		print('Shape of validation_samples:', validation_samples.shape)
		print()

	return train_samples, validation_samples


def process_image(image):
	'''
	Process a given cv2 image in BGR
	Use in both clone.py and drive.py, with exception of RGB or RBG
	Returns the new image
	'''

	image = image[70:-25,:,:]
	image = cv2.GaussianBlur(image, (3,3),0)
	image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)	# cv2 loads images as BGR
	image = image/255.0 - 0.5

	return image


def generator(samples, batch_size=BATCH_SIZE):
	'''
	Load a batch of images at a time and process each of them, including flipping
	Yield: shuffled batch of X_train, y_train to feed to model
	'''
	samples = sklearn.utils.shuffle(samples)
	num_samples = len(samples)
	
	while 1:  # loop so generator doesn't terminate
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			
			# batch_sample[0] = image paths
			# batch_sample[1] = measurements 
			# batch_sample[2] = flip_flags

			for batch_sample in batch_samples:
				
				image = cv2.imread(batch_sample[0])	# Read in image from image path
				angle = float(batch_sample[1])
				if batch_sample[2] == True:			# Flip image if flag is on
					image = cv2.flip(image,1)
					angle = angle*-1.0


				image = process_image(image)


				images.append(image)
				angles.append(angle)

				
			# keras requires numpy arrays, so convert to numpy 
			X_train = np.array(images)
			y_train = np.array(angles)

			# If using grayscale, reshape gray images into tensorflow format...
			#X_train = np.reshape(X_train, (-1,160,320,1)) 
			
			if DEBUGGING:
				print('In Generator...')
				print('Shape of X_train:', X_train.shape)
				print('Shape of y_train:', y_train.shape)

			(X_train, y_train) = sklearn.utils.shuffle(X_train, y_train)
			yield (X_train, y_train)
					
					

####################################
# Main program 
####################################

if __name__ == "__main__":

	
	train_samples, validation_samples = load_data(del_rate=DELETE_RATE, cut_value=MIN_STEERING_ANGLE)

	train_generator = generator(train_samples, batch_size=BATCH_SIZE)
	validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


	# Create Nvidia convnet with 1 output that predicts steering wheel angle based on image 
	# Note: input images are 160x320x3 rgb color, then change colorspace, trimmed, normed
	# Trim top 70 and bottom 25 pixels of image (remove sky and hood of car)...
	# Trimmed image format (160 - 70 off top - 25 off bottom = 65)

	if not DATA_CHECK_ONLY:

		#row, col, ch = 65, 320, 3   
		#row, col, ch = 160, 320, 3
		row, col, ch = 64, 64, 3   # Resize in preprocessing, pass to keras

		model = Sequential()
		#model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(row,col,ch))) # Normalize images
		#model.add(Cropping2D(cropping=((70,25),(0,0))))   
		model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu", input_shape=(row,col,ch)))
		model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
		model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
		model.add(Convolution2D(64,3,3,activation="elu"))
		model.add(Convolution2D(64,3,3,activation="elu"))
		model.add(Flatten())
		model.add(Dense(100))
		model.add(Dense(50))
		model.add(Dense(10))
		model.add(Dense(1))  		# Output is single number - regression


		# Use MSE loss function, not cross entropy, since this is regression, not a classifier
		# Note: Validation error climbed after 5-7 epochs initially, suggesting overfitting...
		# limit epochs. LeNet=5 epochs. Nvidia=3
		model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

		history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
				validation_data=validation_generator, \
				nb_val_samples=len(validation_samples), nb_epoch=4)

		#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)

		print('Saving model...')
		model.save('model.h5')

		# Final plots 
		print_plots(history_object, using_AWS=USING_AWS)


	print()
	print('All done!')








	