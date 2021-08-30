from datetime import datetime

from tensorflow.python.ops.gen_io_ops import save
from utils import parser, save_predictions, read_background
import tensorflow as tf
from tensorflow import keras
from unet import Unet
from dataset import *
import logging
import os


logger = logging.getLogger(__name__)

# Create handlers
f_handler = logging.FileHandler('file.log')
f_handler.setLevel(logging.ERROR)


f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)


if __name__ == '__main__':
    try:
        args = parser()
        print(bool(args.comparison))
        if args.train_image_dir is not None:
            image_dataset = read_files(args.train_image_dir)
            mask_dataset = read_files(args.train_mask_dir)
            full_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
            full_dataset = full_dataset.map(load_image)
            full_dataset = full_dataset.map(normalize)
            full_dataset = full_dataset.map(resize)
            full_dataset = full_dataset.map(augment_data)
            full_dataset = full_dataset.shuffle(args.shuffle_buffer)
            dataset_size = len(full_dataset)
            train_size = int(dataset_size * (1 - args.val_size))
            train_dataset = full_dataset.take(train_size)
            validation_dataset = full_dataset.skip(train_size)
            train_dataset = train_dataset.batch(args.batch_size)
            validation_dataset = validation_dataset.batch(args.batch_size)
            


        if args.test_image_dir is not None:
            image_dataset = read_files(args.test_image_dir)
            mask_dataset = read_files(args.test_mask_dir)
            test_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
            test_dataset = test_dataset.map(load_image)
            test_dataset = test_dataset.map(normalize)
            test_dataset = test_dataset.map(resize_test)
            test_dataset = test_dataset.batch(args.batch_size)

        model = Unet()
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = keras.losses.BinaryCrossentropy()
        miou = keras.metrics.MeanIoU(num_classes=2)
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        checkpoint_filepath = os.path.join(args.checkpoint_path, 'checkpoints')
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
        if args.mode == 'train':
            model.fit(train_dataset, 
                    epochs=args.epochs, 
                    validation_data=validation_dataset, 
                    callbacks=[tensorboard_callback, model_checkpoint_callback])
            

        elif args.mode == 'test' and args.comparison == True:
            tf.config.run_functions_eagerly(True)
            # Load pretrained weights
            model.load_weights(checkpoint_filepath)
            # Set comparison False and mode=test to initiate previous states
            model.comparison = False
            model.mode = 'test'
            # Previous states are now background's feature maps
            background_image = read_background(args.background_image_path)
            model.evaluate(background_image)
            # Set comparison True to compare current image with background
            model.comparison = True
            model.evaluate(test_dataset)
        
        elif args.mode == 'test' and args.comparison == False:
            # Load pretrained weights
            model.load_weights(checkpoint_filepath)
            model.evaluate(test_dataset)
        
        if args.save_predictions and args.mode == 'test':
            save_predictions(model, test_dataset)
    except Exception:
        logger.error("Exception occurred", exc_info=True)


