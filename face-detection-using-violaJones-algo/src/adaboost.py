from functools import partial
import numpy as np
import os

from src.integral_image import IntegralImage as integral
from src.haar_features import HaarLikeFeature as haar
from src.haar_features import featureType
from src.utils import *


# for processing time
import progressbar
from multiprocessing import cpu_count, Pool


LOADING_BAR_LENGTH = 25


def _create_features(img_width, img_height, min_feat_width, max_feat_wid, min_feat_height, max_feat_height):
    # private function to create all possible features
    # return haar_feats: a list of haar-like features
    # return type: np.array(haar.HaarLikeFeature)

    haar_feats = list()

    # iterate according to types of rectangle features
    for feat_type in featureType:

        # min of feature width is set in featureType enum
        feat_start_width = max(min_feat_width, feat_type.value[0])

        # iterate with step 
        for feat_width in range(feat_start_width, max_feat_wid, feat_type.value[0]):

            # min of feature height is set in featureType enum
            feat_start_height = max(min_feat_height, feat_type.value[1])
            
            # iterate with setp
            for feat_height in range(feat_start_height, max_feat_height, feat_type.value[1]):

                # scan the whole image with sliding windows (both vertical & horizontal)
                for i in range(img_width - feat_width):
                    for j in range(img_height - feat_height):
                        haar_feats.append(haar(feat_type, (i,j), feat_width, feat_height, 0, 1)) # threshold = 0 (no misclassified images)
                        haar_feats.append(haar(feat_type, (i,j), feat_width, feat_height, 0, -1)) # threshold = 0 (no misclassified images)

    return haar_feats


def _get_feature_vote(feature, image):
    # para feature: HaarLikeFeature object
    # para image: integral image
    return feature.get_vote(image)


def save_votes(votes):
    np.savetxt("./data/votes.txt", votes, fmt='%f')
    print("...votes saved\n")


def load_votes():
    votes = np.loadtxt("./data/votes.txt", dtype=np.float64)
    return votes


def learn(pos_int_img, neg_int_img, num_classifiers=-1, min_feat_width=1, max_feat_width=-1, min_feat_height=1, max_feat_height=-1, verbose=False):
    # select a set of classifiers, iteratively taking the best classifiers based on a weighted error
    # implementation of AdaBoost algorithm (note pos/1 neg/0)
    '''
    : para pos_int_img, neg_int_img: list of pos/neg integral images
    : type pos_int_img, neg_int_img: list[np.ndarray]
    : para num_classifier: number of classifiers to select, default to use all classifier
    : type num_classifier: int
    : para verbose: whether to print
    : type verbose: boolean
    : return: list of selected features (one classifier has only one feature)
    : rtype: list[haar.HaarLikeFeature]
    '''
    num_pos = len(pos_int_img)
    num_neg = len(neg_int_img)
    num_imgs = num_pos + num_neg
    img_height, img_width = pos_int_img[0].shape

    # maximum features width and height default to image width and height
    # note MAX is noted with 'feature' not 'feat'
    max_feature_width = img_width if max_feat_width == -1 else max_feat_width
    max_feature_height = img_height if max_feat_height == -1 else max_feat_height

    # initialise weights and labels (weights of all image samples)
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos) # w = 1/2m
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg) # w = 1/2l
    weights = np.hstack((pos_weights, neg_weights)) # concatenated weights of images
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg))) # concatenated labels (pos/neg)

    # training images list
    images = pos_int_img + neg_int_img # concatenated image samples

    if verbose:
        print("\ncreating haar-like features ...")
    # all the possible features must be quite time consuming
    features = _create_features(img_width, img_height, min_feat_width, max_feature_width, min_feat_height, max_feature_height)

    if verbose:
        print('... done. %d features were created!' % len(features))

    num_features = len(features)
    feature_index = list(range(num_features)) # save manipulation of data

    # preset number of weak learners (classifiers) [under control]
    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    if verbose:
        print("\ncalculating scores for images ...")

    
    if os.path.exists("./data/votes.txt"):
        votes = load_votes()
    else:
        # 2D numpy.array, each row is an image with all features
        votes = np.zeros((num_imgs, num_features))

        # visualise learning progress with text signals
        bar = progressbar.ProgressBar()
        # pool object to parallelize the execution of a function across multiple input values
        NUM_PROCESS = cpu_count() * 3 # 8 on T580
        pool = Pool(processes=NUM_PROCESS)

        # get all votes for each image and each feature (quite time-consuming)
        for i in bar(range(num_imgs)):
            votes[i, :] = np.array(list(pool.map(partial(_get_feature_vote, image=images[i]), features)))

        save_votes(votes)

    '''
    The partial() is used for partial function application which 'freezes' some portion of a function's arguments and/or keywords
    resulting in a new object with a simplified signature
    In the project, (partial(_get_feature_vote, image=images[i]), features), fixed the argument features
    for i in range(len(features)):
        for j in range(len(images)):
            vote[i][j] = _get_feature_vote(features[i], images[j])
    '''

    # select classifiers
    classifiers = list() # list of HaarLikeFeature objects

    if verbose:
        print("\nselecting classifiers ...")

    # visualise learning progress with text signals
    bar = progressbar.ProgressBar()

    # iterate all classifier (for t = 1, ..., T)
    for _ in bar(range(num_classifiers)):
        
        classification_errors = np.zeros(len(feature_index)) # epsilon_j

        # normalize weights (w_t is a probability distribution) [weights of images]
        weights *= 1. / np.sum(weights)

        # select the best classifier based on the weighted error
        for f in range(len(feature_index)):
            f_idx = feature_index[f]
            # classifier error = sum of misclassified image weights
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            classification_errors[f] = error

        # get the best feature (with the smallest error)
        min_error_idx = np.argmin(classification_errors) 
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]

        # set feature weight (alpha) and add to classifier list
        best_feature = features[best_feature_idx]
        feature_weight = .5 * np.log((1 - best_error) / best_error) # alpha
        best_feature.weight = feature_weight
        classifiers.append(best_feature)

        def new_weights(best_error):
            return np.sqrt((1 - best_error) / best_error)

        # update image weights (w_(t+1) = w_t * beta_t ^ (1 - e_i)), where e_i = 1 when misclassified
        # map(func_to_apply, list_of_inputs) applies a function to all the items in an input_list
        weights_map = map(lambda img_idx: weights[img_idx] * new_weights(best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * 1, range(num_imgs))
        weights = np.array(list(weights_map))

        # remove feature (a feature cannot be selected twice)
        feature_index.remove(best_feature_idx)

    if verbose:
        print("\nclassified selected ...\nreaching the end of AdaBoost algorithm ...")

    return classifiers