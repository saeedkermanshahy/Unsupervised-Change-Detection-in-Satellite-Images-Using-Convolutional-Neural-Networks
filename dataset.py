import tensorflow as tf
import pathlib


def read_files(directory):
    train_dir = pathlib.Path(directory)
    mask_dataset = tf.data.Dataset.list_files(str(train_dir/'*'), shuffle=False)
    return mask_dataset


def load_image(image_path, mask_path):
    mask = tf.io.read_file(mask_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return image, mask


def normalize(image, mask=None):
    image /= 255.
    mask /= 255.
    return image, mask


def resize(image, mask):
    image = tf.image.resize(image, (310, 310),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    mask = tf.image.resize(mask, (310, 310),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image, mask


def resize_test(image, mask):
    image = tf.image.resize(image, (256, 256),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    mask = tf.image.resize(mask, (256, 256),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image, mask


def augment_data(image, mask):
    x = tf.random.uniform(shape=(), minval=1, maxval=200, dtype=tf.int32)
    y = x + 1
    seed = (x, y)
    image = tf.image.stateless_random_crop(image, (256, 256, 3), seed)
    image = tf.image.stateless_random_brightness(image, 0.7, seed)
    image = tf.image.stateless_random_contrast(image, 0.2, 0.7, seed)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    image = tf.image.stateless_random_hue(image, 0.4, seed)
    image = tf.image.stateless_random_saturation(image, 0.3, 1.0, seed=seed)

    mask = tf.image.stateless_random_crop(mask, (256, 256, 1), seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed)
    

    return image, mask



# def load_train_image(image_path, mask_path):
#     image, mask = load_image(image_path, mask_path)
#     image, mask = augment_data(image, mask)
#     image = normalize(image)
#     return image, mask


# def load_test_image(image_path):
#     image = load_image(image_path)
#     image = resize(image, 256, 256)
#     image = normalize(image)
#     return image



if __name__ == "__main__":
    mask_dataset = read_files('/home/qc/Desktop/Share-Tile/sbs/images/train/groundtruth')
    image_dataset = read_files('/home/qc/Desktop/Share-Tile/sbs/images/train/images')
    full_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))



    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))

    full_dataset = full_dataset.map(load_image)
    full_dataset = full_dataset.map(normalize)
    full_dataset = full_dataset.map(resize)
    full_dataset = full_dataset.map(augment_data)
    full_dataset = full_dataset.shuffle(1000)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_dataset = full_dataset.take(val_size)
    validation_dataset = full_dataset.skip(val_size)
    print(len(full_dataset))
    print(len(train_dataset))
    print(len(validation_dataset))
    # full_dataset = full_dataset.batch()

    counter = 1
    for i, j in full_dataset.take(4):
        plt.subplot(4, 2, counter)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(i))
        counter += 1
        plt.subplot(4, 2, counter)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(j))
        plt.axis('off')
        counter += 1
    plt.show()
    
    for p in range(j.shape[0]):
        for q in range(j.shape[1]):
            if j[p, q, 0] == 255:
                print('Ding')