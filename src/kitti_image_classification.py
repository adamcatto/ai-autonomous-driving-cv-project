import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import feature
from scipy.cluster.vq import kmeans, vq
from sklearn.svm import LinearSVC


def image_list(path):
    return sorted([os.path.join(path, file) for file in os.listdir(path)])[1:]


# helper function to construct file header string from int
def pad_int(num, desired_length):
    num_str = str(num)
    num_str_len = len(num_str)
    assert num_str_len <= desired_length
    return '0' * (num_str_len - desired_length) + num_str


# set up data structures to store image info
def structure_image_data(image_index):
    file_num_str = pad_int(num=image_index, desired_length=6)
    with open(train_set[image_index][1], 'r') as label_file:
        metadata = []
        for line in label_file:
            split_info = line.split(' ')
            class_label = split_info[0]
            bbox = split_info[4:8]
            metadata.append((class_label, bbox))
    return metadata


def generate_cropped_images(image_list, label_bbox_database):
    cropped_images = []
    cropped_labels = []
    for key in tqdm(label_bbox_database):
        base_image = Image.open(image_list[key])
        for pair in label_bbox_database[key]:
            label = pair[0]
            if label[0:4] == "Dont":
                continue
            bbox = tuple(int(float(x)) for x in pair[1])
            cropped_image = base_image.crop(bbox).resize((128, 128))
            cropped_images.append(np.array(cropped_image))
            cropped_labels.append(label)
    return cropped_images, cropped_labels


def compute_lbp(image: np.array, radius=1, neighborhood_size=8, method='uniform'):
    if len(image.shape) > 2:
        # convert to grayscale
        image = image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11
    try:
        image_max, image_min = np.max(image), np.min(image)
        image = image - image_min / (image_max - image_min)
    except:
        pass
    lbp = feature.local_binary_pattern(image, neighborhood_size, radius, method)
    #print(lbp)
    #histogram, edges = np.histogram(lbp.ravel(), bins=np.arange(0, neighborhood_size+3), range=neighborhood_size+2)
    #histogram /= np.sum(histogram)
    return lbp


def stack_patterns(pattern_list):
    print('stacking LBPs...')
    s = np.stack(pattern_list)
    print('LBPs stacked, ready to cluster.')
    return s


def generate_bow(stacked_patterns, pattern_list, k=32):
    print('clustering on k=' + str(k) + ' visual words...')
    voc, var = kmeans(stacked_patterns, k, 1)
    print('clustering complete.')
    print(voc, var, voc.shape, var.shape)

    features = np.zeros((len(pattern_list), 64), "float32")
    print('generating features...')
    for i in tqdm(range(len(pattern_list))):
        words, dist = vq(pattern_list[i], voc)
        for w in words:
            features[i][w] += 1
    print('features generated.')
    return features


def train_svm(x_train, y_train):
    classifier = LinearSVC(max_iter=10000)
    classifier.fit(x_train, y_train)
    return classifier


def test_svm(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    correct_predictions = [i for i in range(len(y_test)) if y_test[i] == predicted_classes[i]]
    print('Predicted correctly ' + str(100 * len(correct_predictions) / len(predictions)) + ' percent of the time')
    return predictions, correct_predictions


def naive_bayes():
    pass


def dimensionality_reduction(x):
    pass




    


if __name__ == '__main__':
    raw_image_path = 'data/object-detection/training/image_2/'
    raw_image_files = sorted(os.listdir(raw_image_path))[1:]

    labels_path = 'data/object-detection/labels-final/label_2'
    label_files = sorted(os.listdir(labels_path))[1:]

    raw_images = image_list(raw_image_path)
    #test_images = image_list(test_path)
    label_file_paths = image_list(labels_path)

    image_label_zip = np.array(list(zip(raw_images, label_file_paths)))
    train_size = int(len(image_label_zip) * 0.8)
    images_split = np.split(image_label_zip, [train_size, len(image_label_zip)])
    train_set = images_split[0]
    test_set = images_split[1]

    image_data_structure = {}
    for i in range(len(train_set)):
        image_data_structure[int(pad_int(num=i, desired_length=6))] = structure_image_data(i)

    ######## crop images ########
    print('cropping images...')
    cropped_images, cropped_labels = generate_cropped_images(raw_images, image_data_structure)
    print('images cropped!')
    print('total cropped images: ' + str(len(cropped_images)))
    print(cropped_images[1].shape)


    ######## compute LBP for each image ########
    lbps = [compute_lbp(x) for x in cropped_images]
    cropped_1_lbp = lbps[1]
    print('lbp: ' + str(cropped_1_lbp))
    print('lbp length: ' + str(len(cropped_1_lbp)))
    print('labels example: ' + str(cropped_labels[1]))

    print('labels: ' + str(list(set(cropped_labels))))

    stacked_lbps = stack_patterns(lbps)

    features = generate_bow(stacked_lbps, lbps, k=16)

    classifier = train_svm(cropped_images, cropped_labels)

    cropped_1 = Image.fromarray(cropped_images[1])
    cropped_1.save('cropped_1.png')