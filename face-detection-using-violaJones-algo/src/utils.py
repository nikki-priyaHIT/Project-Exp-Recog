import os
import json
import numpy as np
from PIL import Image
from src.haar_features import featureType
from src.haar_features import HaarLikeFeature as haar
from functools import partial


path_data_face = './train_images/FACES'
path_data_no_face = './train_images/NFACES'
path_json_file = './data/haarClassifiers.json'
path_json_feat = './data/allFeatures.json'
path_cascade = ['./data/cascade_1.json', './data/cascade_2.json', './data/cascade_3.json']
path_classifier = ['./data/classifier_1.json', './data/classifier_2.json', './data/classifier_3.json', 
                    './data/classifier_4.json', './data/classifier_5.json', './data/classifier_6.json',
                    './data/classifier_7.json', './data/classifier_8.json', './data/classifier_9.json',
                    './data/classifier_10.json', './data/classifier_11.json', './data/classifier_12.json',
                    './data/classifier_13.json', './data/classifier_14.json', './data/classifier_15.json',
                    './data/classifier_16.json', './data/classifier_17.json', './data/classifier_18.json',
                    './data/classifier_19.json', './data/classifier_20.json', './data/classifier_21.json']
SIZE = (24, 24) # resize the image


def load_images(face):
    # para face: whether a pos sample or a neg sample (boolean)
    # type face: True or False
    # return para: list of loaded images

    path = path_data_face if face else path_data_no_face
    images = list()
    # all files in the directory
    for _file in os.listdir(path):
        # not a directory
        if not os.path.isdir(_file):
            # is a bmp image file
            if _file.endswith('.bmp'):
                # loading images
                img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
                img_arr.resize(SIZE)
                images.append(img_arr)

    return images


def images_partition(images):
    # para images: list of images read for partition
    # return para: three partitions of images
    # partition ratio: train/dev/test = 0.8/0.1/0.1
    num_img = len(images)
    tenth_num = int(num_img / 10)
    train_num = num_img - 2 * tenth_num
    train_imgs = images[:train_num]
    dev_imgs = images[train_num:train_num + tenth_num]
    test_imgs = images[train_num + tenth_num:]
    return train_imgs, dev_imgs, test_imgs


def write_text_file(FP_rate, FN_rate, num_classifier):
    # write FP rates and FN rates to text file
    f = open('./results_FP_FN.txt', 'a+', encoding='utf-8')
    f.write("FP RATE: %f\tFN RATE: %f\tnumber of classifiers: %d\n" % (FP_rate, FN_rate, num_classifier))
    f.close()


def write_json_file(classifiers, classifier, cascade, n_classifier=0, no_cascade=0):
    # write the haar-like classifiers into a json file
    # para classifiers: list of classifiers/features
    # type classifiers: list[haar.HaarLikeFeature]
    classifiers_dict = dict()

    path = path_cascade[no_cascade] if cascade else path_json_file
    path = path_classifier[no_cascade] if classifier else path_json_file
    
    with open(path, 'w') as fp:
        i = 0
        for c in classifiers:
            c_dict = dict()
            c_dict["feature type"] = c.type.name
            c_dict["position"] = c.top_left
            c_dict["width"] = c.width
            c_dict["height"] = c.height
            c_dict["threshold"] = c.threshold
            c_dict["parity"] = c.parity
            c_dict["weight"] = c.weight
            classifiers_dict[str(i)] = c_dict
            i += 1
        json.dump(classifiers_dict, fp, indent=4)
    

def load_json_file():
    # load the haar-like classifiers from a json file
    # return: list of classifiers/features
    # rtype: list[haar.HaarLikeFeature]
    classifiers = list()
    with open(path_json_file, 'r') as fp:
        data_list = json.load(fp)
        for i in range(len(data_list)):
            data = data_list[str(i)]
            classifiers.append(haar(data["feature type"], data["position"], data["width"], data["height"], data["threshold"], data["parity"], data["weight"]))
    return classifiers


def reconstruct(classifiers, img_size):
    # create an image by putting all given classifiers on top of each other producing an archetype of the learned class of object
    '''
    : para classifiers: one classifier/feature
    : type classifiers: haar.HaarLikeFeature
    : para img_size: tuple of width and height
    : type img_size: (int, int)
    : return: reconstructed image
    : rtype: PIL.Image
    '''
    image = np.zeros(img_size)

    for c in classifiers:
        # map parity: -1 -> 0, 1 -> 1
        parity = pow(1 + c.parity, 2)/4

        if c.type == "TWO_VERTICAL":
            for x in range(c.width):
                sign = parity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight

        elif c.type == "TWO_HORIZONTAL":
            sign = parity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight

        elif c.type == "THREE_HORIZONTAL":
            sign = parity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight

        elif c.type == "THREE_VERTICAL":
            for x in range(c.width):
                sign = parity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight

        elif c.type == "FOUR":
            sign = parity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight

    image -= image.min()
    image /= image.max()
    image *= 255

    # reverse the color
    image *= -1
    image += 255
    # expand the size
    result = Image.fromarray(image.astype(np.uint8))
    
    return result


def ensemble_vote(int_img, classifiers):
    # classify given integral image (numpy array) using given classifiers
    # i.e. if the sum of all classifier votes is greater than 0, image is classified pos/1 else neg/0
    # the threshold is 0, because votes can be +1 or -1
    '''
    : para int_img: integral image to be classified
    : type int_img: np.ndarray
    : para classifier: list of classifiers
    : type classifier: list[haar.HaarLikeFeature]
    : return: 1 iff sum of classifier votes is greater than 0, else 0
    : rtype: int
    '''
    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):
    # classify given list of integral images using given classifiers
    '''
    : para int_imgs: list of integral images to be classified
    : type int_imgs: list[np.ndarray]
    : para classifier: list of classifiers
    : type classifier: list[haar.HaarLikeFeature]
    : return: list of assigned labels, 1 iff sum of classifier votes is greater than 0, else 0
    : rtype: list[int]
    '''
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def count_Rate(pos_images, neg_images, classifiers):
    # count the FalsePositiveRate and FalseNegativeRate of one strong classifier generated by AdaBoost
    # para classifiers: classifiers of specific size 
    # type classifiers: list[haar.HaarLikeFeature]
    # para images: list of images ready for prediction
    # return: FR rate and FN rate
    num_pos = len(pos_images)
    num_neg = len(neg_images)

    # True positives
    correct_pos_image = sum(ensemble_vote_all(pos_images, classifiers))
    # False negatives (vote neg for pos images)
    incorrect_pos_image = num_pos - correct_pos_image
    
    # False positives (vote pos for neg images)
    incorrect_neg_image = sum(ensemble_vote_all(neg_images, classifiers))
    # True negatives
    correct_neg_image = num_neg - incorrect_neg_image

    TP, FN, FP, TN = correct_pos_image, incorrect_pos_image, incorrect_neg_image, correct_neg_image

    return TP, FN, FP, TN


def two_haar_equal(haar0, haar1):
    # check whether two haar like features are the same 
    # para haar0: haar features ready to write a json file
    # para haar1: haar features loaded from the json file
    if str(haar0.type.name) != str(haar1.type):
        return False
    elif haar0.top_left[0] != haar1.top_left[0] or haar0.top_left[1] != haar1.top_left[1]:
        return False
    elif haar0.width != haar1.width:
        return False
    elif haar0.height != haar1.height:
        return False
    elif haar0.threshold != haar1.threshold:
        return False
    elif haar0.parity != haar1.parity:
        return False
    elif int(haar0.weight) != int(haar1.weight):
        return False
    else:
        return True
    