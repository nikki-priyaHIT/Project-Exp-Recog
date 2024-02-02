import src.integral_image as integral
import src.haar_features as haar
from src.haar_features import featureType
import src.adaboost as ab
import src.cascaded as cas
import src.utils as utils
import numpy as np

import progressbar



if __name__ == "__main__":

    pos_image_path = './train_images/NFACES'
    neg_image_path = './train_images/NFACES'

    # load images
    print("\nloading image samples from files ...")
    pos_images = utils.load_images(True)
    neg_images = utils.load_images(False)
    print("\nthe number of pos samples loaded: %d\nthe number of neg samples loaded: %d" % 
        (len(pos_images), len(neg_images)))

    # images partition
    print("\npartitioning images into train/dev/test sets ...")
    pos_train_imgs, pos_dev_imgs, pos_test_imgs = utils.images_partition(pos_images)
    neg_train_imgs, neg_dev_imgs, neg_test_imgs = utils.images_partition(neg_images)
    num_train_pos, num_dev_pos, num_test_pos = len(pos_train_imgs), len(pos_dev_imgs), len(pos_test_imgs)
    num_train_neg, num_dev_neg, num_test_neg = len(neg_train_imgs), len(neg_dev_imgs), len(neg_test_imgs)
    print("\nPOSITIVE IMAGE SAMPLE\nthe number of training set: %d\nthe number of development set: %d\nthe number of test set: %d" % 
        (num_train_pos, num_dev_pos, num_test_pos))
    print("\nNEGATIVE IMAGE SAMPLE\nthe number of training set: %d\nthe number of development set: %d\nthe number of test set: %d" % 
        (num_train_neg, num_dev_neg, num_test_neg))

    # integral images
    pos_train_int_imgs, neg_train_int_imgs = list(), list()
    pos_dev_int_imgs, neg_dev_int_imgs = list(), list()
    pos_test_int_imgs, neg_test_int_imgs = list(), list()
    pos_train_variance, neg_train_variance = list(), list()

    bar = progressbar.ProgressBar()

    print("\ngetting integral images ...")
    for i in range(num_train_pos):
        int_img_pos, var_pos = integral.IntegralImage(pos_train_imgs[i]).get_integral_image()
        pos_train_int_imgs.append(int_img_pos)
        pos_train_variance.append(var_pos)

    for j in range(num_train_neg):
        int_img_neg, var_neg = integral.IntegralImage(neg_train_imgs[j]).get_integral_image()
        neg_train_int_imgs.append(int_img_neg)
        neg_train_variance.append(var_neg)

    for k in range(num_dev_pos):
        int_img_pos_dev, var_pos_dev = integral.IntegralImage(pos_dev_imgs[k]).get_integral_image()
        int_img_pos_test, var_pos_test = integral.IntegralImage(pos_test_imgs[k]).get_integral_image()
    
        pos_dev_int_imgs.append(int_img_pos_dev)
        pos_test_int_imgs.append(int_img_pos_test)
        
    for l in range(num_dev_neg):
        int_img_neg_test, var_neg_test = integral.IntegralImage(neg_test_imgs[l]).get_integral_image()
        int_img_neg_dev, var_neg_dev = integral.IntegralImage(neg_dev_imgs[l]).get_integral_image()
        neg_dev_int_imgs.append(int_img_neg_dev)
        neg_test_int_imgs.append(int_img_neg_test)

    print("\nintegral images obtained")

    # get the variance right (not integrated yet)
    # train_variance = pos_train_variance + neg_train_variance

    # # parameters (do not change)
    num_classifier = 10
    min_feature_height = 4
    max_feature_height = 10
    min_feature_width = 4
    max_feature_width = 10

    print("\nAdaBoost begins ...")
    classifiers = ab.learn(pos_train_int_imgs, neg_train_int_imgs, num_classifier, 
        min_feature_width, max_feature_width, min_feature_height, max_feature_height, verbose=True)
    
    utils.write_json_file(classifiers, False, False, num_classifier, 0)


    # utils.write_json_file(classifiers)

    cascade = cas.cascaded_classifier(pos_train_int_imgs, neg_train_int_imgs, pos_dev_int_imgs, neg_dev_int_imgs)

    cas.test_cascade_classifier(pos_test_int_imgs, neg_test_int_imgs, cascade)

