# Unsupervised-Change-Detection-in-Satellite-Images-Using-Convolutional-Neural-Networks

## Usage

```python
# to train
python main.py --mode train --train_image_dir [dir of train images] --train_mask_dir [dir of train masks] --checkpoint_path [dir for saving checkpoints] --epochs [number of epochs for training default=50] --learning_rate [learning rate for optimization default=0.001] --batch_size [number of image samples batched together default=32] --val_size [ratio of validation dataset default=0.2] --shuffle_buffer [shuffle buffer size for shuffle funtion default=1000] 

# to test with no comparison
python main.py --mode test --test_image_dir [dir of test images] --test_mask_dir [dir of test masks] --checkpoint_path [dir for loading checkpoints] --batch_size [set it 1 if you want save predictions] --save_predictions [boolean]

# to test with comparison
python main.py --mode test --comparison True --test_image_dir [dir of test images] --test_mask_dir [dir of test masks] --checkpoint_path [dir for loading checkpoints] --batch_size [set it 1 if you want save predictions] --background_image_path [path of background image for comparison] --save_predictions [boolean]

```
