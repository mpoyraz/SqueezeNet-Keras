
import os, argparse
import utils, squeezeNet
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD


def train(train_dir, mean_image_path, batchsize, num_epochs, 
          lr, weight_decay_l2, img_height, img_width):
    """
    Trains the model with given batchsize for given number of epochs.
    Subtracts the mean image for data centralization.
    Data augmentation is done with Keras ImageDataGenerator.
    
    # Arguments
        train_dir: directory for training/validation images
        mean_image_path: full path for the mean image
        batchsize: size of the batch for training
        num_epochs: number of epocs to run training
        lr: initial learning rate 
        weight_decay_l2: weight decay for conv layers
        img_height: columns of the images
        img_width: rows of the images
        
    # Return
        model: trained keras model
        model_parms: dictionary of model parameters (classes and image dimensions)
        train_history: training history of loss, accuracy and learning rate
        
    """
    # Make './model' directory to store trained model and model parameters
    if not os.path.exists('./model'):
        os.makedirs('./model')
    
    # Data augmentation
    datagen = ImageDataGenerator(featurewise_center=True, samplewise_center=False, 
                                 featurewise_std_normalization=False, 
                                 samplewise_std_normalization=False, 
                                 zca_whitening=False, zca_epsilon=1e-06, 
                                 rotation_range=20, width_shift_range=0.1, 
                                 height_shift_range=0.1, brightness_range=None, 
                                 shear_range=0.01, zoom_range=0.1, 
                                 channel_shift_range=0.0, fill_mode='nearest',
                                 cval=0.0, horizontal_flip=True, vertical_flip=False, 
                                 rescale=None, preprocessing_function=None, 
                                 data_format="channels_last", validation_split=0.1, dtype=None)
    
    # Mean image of the dataset
    img_mean_array = img_to_array(load_img(mean_image_path, target_size=(img_height, img_width)))
    datagen.mean = img_mean_array
    
    # Train generator
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batchsize,
        class_mode='categorical',
        subset='training', # set as training data
        interpolation='bilinear')
    
    # Validation generator
    validation_generator = datagen.flow_from_directory(
        train_dir, # same directory as training data
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batchsize,
        class_mode='categorical',
        subset='validation', # set as validation data
        interpolation='bilinear')
    
    classes = list(train_generator.class_indices.keys())
    num_classes = len(classes)
    
    # SGD Optimizer
    opt = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
    
    # Generate and compile the model
    model = squeezeNet.SqueezeNet(num_classes, weight_decay_l2, inputs=(img_height, img_width, 3))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Steps per epoch
    step_size_train=train_generator.n//train_generator.batch_size
    step_size_valid=validation_generator.n//validation_generator.batch_size
    
    step_size_train=5
    step_size_valid=2
    
    # Linear LR decay after each batch update
    linearDecayLR = utils.LinearDecayLR(min_lr=1e-5, max_lr=lr, 
                                        steps_per_epoch=step_size_train, 
                                        epochs=num_epochs, verbose=1)
    
    # Train the model
    print("Start training the model")
    training_history = model.fit_generator(
        	train_generator,
        	steps_per_epoch=step_size_train,
        	validation_data=validation_generator,
        	validation_steps=step_size_valid,
        	epochs=num_epochs,
            verbose=1, 
            callbacks=[linearDecayLR])
    print("Model training finished")
    
    # Model parameters to be used for prediction
    model_parms = {'num_classes': num_classes,
                   'classes': classes,
                   'height': img_height,
                   'width': img_width}
    
    # Training history
    train_history = training_history.history
    
    return model, model_parms, train_history
    
if __name__ == "__main__":
    # Parse arguements
    parser = argparse.ArgumentParser(description="SqueezeNet Training.")

    parser.add_argument("--dir", type=str, default='./train',
                        help="Directory for training/validation images.")
    parser.add_argument("--mean-image", type=str, default='./images/mean_image.jpg', dest='mean_image',
                        help="Mean image for training dataset.")
    parser.add_argument("--batchsize", type=int, default=64,
                        help="Size of the batch for training, default: 64.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs, default: 20.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate of SGD with momentum, default 0.01.")
    parser.add_argument("--weight-decay-l2", type=float, default=0.0001, dest='weight_decay_l2',
                        help="Weight decay for conv layers, default 0.0001.")
    parser.add_argument("--img-width", type=int, default=128, dest='width',
                        help="Rows of the images, default: 128.")
    parser.add_argument("--img-height", type=int, default=128, dest='height',
                        help="Columns of the images, default: 128.")
    args = parser.parse_args()

    # Train the model
    model, model_parms, training_history = train(args.dir, args.mean_image, 
                                                 args.batchsize, args.epochs, 
                                                 args.lr, args.weight_decay_l2,
                                                 args.width, args.height)
    # Save the trained model
    model.save('./model/squeezenet_model.h5')
    print("Trained Squeezenet Keras model is saved")
    
    # Save the model parms for prediction
    utils.save_model_parms(model_parms, fname='./model/model_parms.json')
    print("Model parameters (classes, image size) are saved")
    
    # Save the training history for train/val loss and accuracy
    utils.save_training_history(training_history, fname='./model/training_history.npy')
    print("Training history of loss, accuracy and learning rate is saved")
    
    # Plot the training history
    utils.plot_training_history(training_history)
    