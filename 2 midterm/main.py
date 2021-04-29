import random

import cv2
import gensim
import matplotlib.pyplot as plt
import numpy as np
import time

import pylab as p
from sklearn.cluster import KMeans
from pathlib import Path
import copy

from gensim.models import ldamodel


def sift_descriptors(path, drawKeypoints=False):
    image = cv2.imread(path)
    sift = cv2.xfeatures2d.SIFT_create()
    # detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(image, None)
    if (drawKeypoints):
        cv2.drawKeypoints(image, kp, image)
        cv2.imshow("keypoints", image)
        cv2.waitKey(0)
    return kp, des


def mser_descriptors(path, drawKeypoints=False):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # detect keypoint using MSER
    mser = cv2.MSER_create(_max_area=4000)
    keypoints = mser.detect(image)
    descriptors = sift.compute(image, keypoints)[1]

    if (drawKeypoints):
        cv2.drawKeypoints(image, keypoints, image, color=(255, 0, 0))
        cv2.imshow("keypoints", image)
        cv2.waitKey(0)
    return keypoints, descriptors


def orb_descriptors(path, drawKeypoints=False):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # detect keypoint using MSER
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    descriptors = sift.compute(image, keypoints)[1]

    if (drawKeypoints):
        cv2.drawKeypoints(image, keypoints, image, color=(255, 0, 0))
        cv2.imshow("keypoints", image)
        cv2.waitKey(0)
    return keypoints, descriptors


def run_mser(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()

    mser = cv2.MSER_create(_min_area=200, _max_area=4000)

    regions, _ = mser.detectRegions(img)
    keypoints = mser.detect(img)
    sizes = [s.size for s in keypoints]
    print(sizes)
    print(len(regions), len(keypoints))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.drawKeypoints(vis, keypoints, vis, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', vis)
    cv2.waitKey(0)


def run_orb(image_path):
    img = cv2.imread(image_path, 0)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)

    return kp


def kmeans(descriptors, n_clusters=500):
    # init k means
    kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10)
    # run kmeans (fit() function), and return the labels which are the codes of the clusters
    # assigned to each point
    fitted_kmeans = kmeans.fit(descriptors)
    return fitted_kmeans, fitted_kmeans.labels_


def kmeans_predict(kmeans, descriptors):
    # use the predict method of the kmeans obj
    return kmeans.predict(descriptors)


def create_BoW_1():
    assigned_clusters = kmeans(des, n_clusters=n_clusters)
    # create the hist of the Bag of Words
    BoW, _ = np.histogram(assigned_clusters, bins=n_clusters)
    # plt.hist(assigned_clusters, bins=n_clusters)
    # plt.show()


def get_topic(word_topics, searched_word):
    for word in word_topics:
        if word[0] == searched_word:
            if len(word[1]) > 0:
                return word[1][0]
            else:
                return None


def check_colors_distance(colors, checking_color, distance_thr):
    for color in colors:
        distance = sum([abs(color[i] - checking_color[i]) for i in range(0, 3)])
        if distance < distance_thr:
            return False
    return True


def create_topic_colors(n_topics, distance_thr=120):
    result = []
    for n in range(n_topics):
        while (True):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            if (check_colors_distance(result, (r, g, b), distance_thr)):
                break
        result.append((r, g, b))
    return result


def create_colors_image(topic_colors):
    # Create a black image
    img_height = 200
    img_width = 1000
    img = np.zeros((img_height, img_width, 3), np.uint8)
    rect_length = img_width // len(topic_colors)

    for i in range(1, len(topic_colors) + 1):
        cv2.rectangle(img, ((i - 1) * (rect_length), 0), (i * rect_length, img_height - 1), topic_colors[i - 1], -1)

    return img


def get_rect_id(keypoint, image):
    # to judge a point(x,y) is in the rectangle, just to check if a < x < a+c and b < y < b + d
    image_object = cv2.imread(image['path'].__str__())
    h, w, image_channels = image_object.shape

    #segments_mask = [(0,0,w//2, h//2), (w//2+1, 0, w//2, h//2),
    #                 (0, h//2+1, w//2, h//2), (w//2+1, h//2+1, w//2, h//2)]

    segments_mask = [(0,0,w//4,h//4), (w//4,0,w//2,h//4), (w//4*3,0,w-w//4*3, h//4),
                     (0,h//4,w//4,h//2), (w//4, h//4, w//2, h//2), (w//4*3, h//4, w-w//4*3, h//2),
                     (0, h//4*3, w//4, h-h//4*3), (w//4, h//4*3, w//2, h-h//4*3), (w//4*3, h//4*3, w-w//4*3, h-h//4*3)]

    image_with_mask = cv2.imread(image['path'].__str__())
    for rect in segments_mask:
        image_with_mask = cv2.rectangle(image_with_mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255,255,255))

    #cv2.imshow("image mask", image_with_mask)
    #cv2.waitKey(0)

    x, y = keypoint.pt
    x, y = int(x), int(y)
    for rect_id, rect in enumerate(segments_mask):
        a, b, c, d = rect
        if x >= a and x < a + c and y >= b and y < b + d:
            return rect_id


def compute_segments(image, n_clusters):
    segments = [{'assigned_clusters': []} for x in range(0,9)]
    for keypoint_idx, keypoint in enumerate(image['keypoints']):
        assigned_cluster = image['assigned_clusters'][keypoint_idx]
        rect_id = get_rect_id(keypoint, image)
        segments[rect_id]['assigned_clusters'].append(assigned_cluster)

    for segment in segments:
        segment['hist'] = [0 for x in range(0, n_clusters)]
        for cluster in segment['assigned_clusters']:
            segment['hist'][cluster] += 1
        segment['frequencies'] = [(i, value) for i, value in enumerate(segment['hist'])]

    return segments



if __name__ == '__main__':
    # config
    n_clusters = 500
    n_topics = 6
    alpha = 0.001
    passes = 2
    iterations = 100
    save_option = False

    start_time = time.time()

    # splitting the dataset into training and test
    # construct the validator, that will allow to include images in the set
    training_set_validator = []
    test_set_validator = []
    for y in range(1, 3):
        test_data_1 = random.randint(1, 30)
        test_data_2 = test_data_1
        while test_data_2 == test_data_1:
            test_data_2 = random.randint(1, 30)

        test_set_validator.append("{}_{}_s.bmp".format(y, test_data_1))
        test_set_validator.append("{}_{}_s.bmp".format(y, test_data_2))

        for x in range(1, 31):
            if x == test_data_1 or x == test_data_2:
                continue
            training_set_validator.append("{}_{}_s.bmp".format(y, x))

    print("Test set validator", test_set_validator)
    print("len:", len(test_set_validator))
    print("Training set validator", training_set_validator)
    print("len:", len(training_set_validator))

    # create the training set
    training_set = [{"path": path, "filename": path.name} for path in Path("images/").rglob('*') if
                    path.name in training_set_validator]

    # create the test set
    test_set = [{"path": path, "filename": path.name} for path in Path("images/").rglob('*') if
                path.name in test_set_validator]

    # extract descriptors from all the images (training + test)
    for image in training_set + test_set:
        kp, des = mser_descriptors(image['path'].__str__(), drawKeypoints=False)
        # orb_kp, orb_des = orb_descriptors(image['path'].__str__())

        # kp = np.concatenate((mser_kp,orb_kp))
        # des = np.concatenate((mser_des, orb_des))

        # testing diff between features detectors
        '''
        out_image = cv2.imread(image['path'].__str__())
        out_image_2 = cv2.imread(image['path'].__str__())
        cv2.drawKeypoints(out_image,kp,out_image)
        cv2.drawKeypoints(out_image_2,orb_kp,out_image_2)
        cv2.imshow("MSER", out_image)
        cv2.imshow("ORB", out_image_2)
        cv2.waitKey(0)
        '''

        image['descriptors'] = copy.copy(des)
        image['keypoints'] = copy.copy(kp)

    # get all descriptors of the training set
    all_training_descriptors = np.concatenate([image['descriptors'] for image in training_set])
    print("Keypoints extracted! Training keypoints extracted:", len(all_training_descriptors))

    print("Running KMeans..")
    # run kmeans on the training set
    # the function return th kmeans object
    # and the cluster code to which every descriptor is assigned
    kmeans_obj, assigned_clusters = kmeans(all_training_descriptors, n_clusters=n_clusters)
    print("Kmeans completed\nCreating images data structures..")
    # create BoW for each image in the training set (array of tuples (cluster_code, # descriptors))
    slot_start_index = 0
    for image in training_set:
        image['assigned_clusters'] = assigned_clusters[
                                     slot_start_index:slot_start_index + image['descriptors'].shape[0]]
        slot_start_index += image['descriptors'].shape[0]

        #image['hist'], _ = np.histogram(image['assigned_clusters'], bins=n_clusters)

        image['hist'] = [0 for x in range(0,n_clusters)]
        for cluster in image['assigned_clusters']:
            image['hist'][cluster] += 1

        image['frequencies'] = [(i, value) for i, value in enumerate(image['hist'])]

        segments = compute_segments(image, n_clusters)
        image['segments'] = segments
        image['segments_freqs'] = [segment['frequencies'] for segment in segments]

    '''
    print("Testing Kmenas algorithm..")
    #test kmeans on features
    print("Creating colors for clusters detected..")
    features_colors = create_topic_colors(len(np.unique(image['assigned_clusters'])), distance_thr=1)
    features_colors_dict = {}
    features_colors_image = create_colors_image(features_colors)
    cv2.imshow("colors", features_colors_image)
    print("Colors: {}".format(features_colors))
    prev_subset = 0
    for image in training_set:
        if image['filename'][0] == prev_subset:
            continue
        prev_subset = image['filename'][0]
        print ("image {}".format(image['filename']))
        output_image=cv2.imread(image['path'].__str__())
        print("Assigning colors")
        for idx,element in enumerate (np.unique(image['assigned_clusters'])):
            features_colors_dict[element] = features_colors[idx]
        print("Assignments: {}".format(features_colors_dict))
        print("Drawing keypoints..")
        for i, keypoint in enumerate(image['keypoints']):
            cv2.drawKeypoints(output_image,[keypoint],output_image ,color=features_colors_dict[image['assigned_clusters'][i]],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("{}_kmeans result".format(image['filename'].__str__()), output_image)
    cv2.waitKey(0)
    '''

    # create BoW for each image in the test set
    for image in test_set:
        image['assigned_clusters'] = kmeans_predict(kmeans_obj, image['descriptors'])

        #image['hist'], _ = np.histogram(image['assigned_clusters'], bins=n_clusters)
        image['hist'] = [0 for x in range(0, n_clusters)]
        for cluster in image['assigned_clusters']:
            image['hist'][cluster] += 1

        image['frequencies'] = [(i, value) for i, value in enumerate(image['hist'])]

        segments = compute_segments(image, n_clusters)
        image['segments'] = segments
        image['segments_freqs'] = [segment['frequencies'] for segment in segments]

    print("Init LDA Model..")


    # init LDA model
    t = []
    for image in training_set:
        for segment in image['segments_freqs']:
            t.append(segment)

    print(np.array(t).shape)
    print(np.array([image['frequencies'] for image in training_set]).shape)
    #lda_model = ldamodel.LdaModel([image['frequencies'] for image in training_set], num_topics=n_topics,
    #                              alpha=alpha, per_word_topics=True, passes=passes, iterations=iterations)

    lda_model = ldamodel.LdaModel(t, num_topics=n_topics,
                                  alpha=alpha, per_word_topics=True, passes=passes, iterations=iterations)

    print("Creating color for each topic")
    # create colors palette
    topic_colors = create_topic_colors(n_topics)
    colors_image = create_colors_image(topic_colors)
    cv2.imshow("colors", colors_image)
    if save_option:
        # colors_output_name = "output/a{}_t{}_c{}_p{}_i{}_".format(alpha, n_topics, n_clusters, passes, iterations)
        colors_output_name = "output/all_dc/c500-t12-a1-p2-i100/test3_"
        colors_output_name = colors_output_name + "COLORS.bmp"
        cv2.imwrite(colors_output_name, colors_image)
        cv2.waitKey(0)

    print("Time elapsed: {}s".format(round(time.time() - start_time, 2)))

    images_topics_distributions = {}
    for image in test_set:
        # get document topics from LDS model
        # set per_word_topics = True in order to get also topic distribution for each word
        image_topics, word_topics, phi_values = lda_model.get_document_topics(image['frequencies'],
                                                                              per_word_topics=True)
        # image_topics, word_topics, phi_values = lda_model[image['frequencies']]

        print(image['filename'], image_topics)
        print(image['filename'], word_topics)

        # saving the image distribution
        image_topics_distribution = [0 for x in range(0, n_topics)]
        for prob in image_topics:
            image_topics_distribution[prob[0]] = prob[1]
        images_topics_distributions[image['filename']] = image_topics_distribution

        output_image = cv2.imread(image['path'].__str__())
        output_image_all_k = cv2.imread(image['path'].__str__())

        for i, keypoint in enumerate(image['keypoints']):
            word = image['assigned_clusters'][i]
            # take the topic with higher probability
            topic = get_topic(word_topics, word)

            if topic is None:
                continue
            # draw keypoints
            # cv2.drawKeypoints(output_image, [keypoint], output_image, color=topic_colors[topic],
            #                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(output_image, [keypoint], output_image, color=topic_colors[topic])

        #cv2.drawKeypoints(output_image_all_k, image['keypoints'], output_image_all_k, color=(255, 255, 255))

        cv2.imshow(image['filename'].__str__(), output_image)
        #cv2.imshow(image['filename'].__str__() + "all", output_image_all_k)
        if save_option:
            # output_name = "output/a{}_t{}_c{}_p{}_i{}_".format(alpha, n_topics, n_clusters, passes, iterations)
            output_name = colors_output_name
            output_name = output_name + image['filename'].__str__()
            print("Saving", output_name)
            cv2.imwrite(output_name, output_image)
    cv2.waitKey(0)

    for distribution in images_topics_distributions.items():
        distribution_min = (None, np.inf, None)
        for testing_brother in images_topics_distributions.items():
            if testing_brother[0] != distribution[0]:
                diff = np.linalg.norm(np.array(distribution[1]) - np.array(testing_brother[1]))
                if diff < distribution_min[1]:
                    distribution_min = (testing_brother[0], diff, testing_brother[1])

        print(distribution[0], distribution[1], distribution_min[0], distribution_min[2], distribution_min[1])
