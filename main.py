from sys import stdout
import os
import glob
import cv2
import numpy as np

images = glob.glob(r'/Users/kwondongjae/Library/CloudStorage/OneDrive-개인/멀티미디어/HW5/roadimage/*.jpg')
querys = glob.glob(r'/Users/kwondongjae/Library/CloudStorage/OneDrive-개인/멀티미디어/HW5/query/query*.jpg')


def rename_FILENAME(IMAGE_NAME):
    find_slash = IMAGE_NAME.rfind('/')
    find_extension = IMAGE_NAME.rfind('.')
    if find_slash == -1:
        print("[ERROR] 경로 위치가 없습니다.")
    elif find_extension == -1:
        print("[ERROR] 파일명에서 확장자를 찾을 수 없습니다.")
    else:
        imgNum_name = IMAGE_NAME[find_slash + 1:find_extension] + ".png"
        return imgNum_name


def print_progressing(progress):
    # Initialize
    str_progress = "["
    epoch = 25
    # progress(%) -> [====...---]
    progress_now = (progress / 100) * epoch
    for index in range(epoch):
        if index < int(progress_now):
            str_progress += "="
        else:
            str_progress += "-"
    str_progress += "]"
    # Print Present Progress
    stdout.write("\r%9.5f%% completed %s" % (progress, str_progress))
    stdout.flush()


# SIFT Algorithms
class SIFT_Algorithms:
    def __init__(self, pw=0, IMAGE_list=[], IMAGE_FILENAME_list=[]):
        self.image_list = IMAGE_list
        self.image_name_list = IMAGE_FILENAME_list
        self.SIFT_Image = []
        self.kp_list = []
        self.mp_list = []
        self.good_test_image_filename = ""
        self.progress = 0
        self.pw = pw        # progress interval(width)

    def find_Matching_point(self, image):
        good_feature_cnt = 0
        output_kp = ()
        output_kp_train = ()
        sift = cv2.SIFT_create()

        print_progressing(self.progress)
        for index, imgNum in enumerate(self.image_list):
            self.progress += 1 / len(self.image_list) * self.pw
            kpI, desI = sift.detectAndCompute(image, None)
            kpC, desC = sift.detectAndCompute(imgNum, None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(desI, desC, k=2)
            # matchesMask = [[1순위 매칭, 2순위 매칭],...]
            matchesMask = [[0, 0] for _ in range(len(matches))]

            good_feature = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.5 * n.distance:
                    matchesMask[i] = [1, 0]
                    good_feature += 1

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(0, 0, 255),
                               matchesMask=matchesMask,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # dst = cv2.drawMatchesKnn(image, output_kp, imgNum, output_kp_train, matches, None, **draw_params)
            if good_feature_cnt < good_feature:
                good_feature_cnt = good_feature
                self.good_test_image_filename = self.image_name_list[index]
                output_kp = kpI
                output_kp_train = kpC
                self.SIFT_Image = cv2.drawMatchesKnn(image, output_kp, imgNum, output_kp_train,
                                                     matches, None, **draw_params)
                self.SIFT_Image = np.array(self.SIFT_Image, dtype=np.float64)

            print_progressing(self.progress)

        return self.SIFT_Image, good_feature_cnt, output_kp, output_kp_train, self.good_test_image_filename

    def save_MAX_KP(self, image, kp, filename=""):
        self.kp_list.append([image, kp, filename])

        if len(self.kp_list) > 1:
            for max_iter in range(len(self.kp_list)-1, -1, -1):
                for index in range(max_iter):
                    if len(self.kp_list[index][1]) > len(self.kp_list[index + 1][1]):
                        self.kp_list[index], self.kp_list[index + 1] = self.kp_list[index + 1], self.kp_list[index]
            # Max Num is 10
            while len(self.kp_list) > 3:
                self.kp_list.pop(0)

    def save_MAX_MP(self, image, kp1, kp2, filename="", test_filename="", cnt=0):
        self.mp_list.append([image, filename, test_filename, cnt, kp1, kp2])

        if len(self.mp_list) > 1:
            for max_iter in range(len(self.mp_list)-1, -1, -1):
                for index in range(max_iter):
                    if self.mp_list[index][3] > self.mp_list[index + 1][3]:
                        self.mp_list[index], self.mp_list[index + 1] = self.mp_list[index + 1], self.mp_list[index]
            # Max Num is 10
            while len(self.mp_list) > 3:
                self.mp_list.pop(0)

    # Run Function
    def run_SIFT_function(self, image, filename=""):
        # Update List
        matching_image, feature_num, kp1, kp2, test_image_name = self.find_Matching_point(image)
        self.save_MAX_KP(image, kp1, filename=filename)
        self.save_MAX_MP(matching_image, kp1, kp2, filename=filename,
                         test_filename=test_image_name, cnt=feature_num)

    def save_MAX3KP_image(self):
        print("Start Upload Maximum Keypoint Image")
        for index, IMAGE_INFO in enumerate(self.kp_list):
            MAXKP_IMAGE_FILENAME = "Max Keypoint(" + str(3-index) + ").png"
            print("Upload Maximum-Matching-Point Image : ", MAXKP_IMAGE_FILENAME)
            dst = cv2.drawKeypoints(IMAGE_INFO[0], IMAGE_INFO[1], None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.imwrite(MAXKP_IMAGE_FILENAME, dst, [cv2.IMWRITE_PNG_COMPRESSION])
        print("Finish!!")

    def print_MAX3MP_INFO(self):
        for index, SIFT_IMAGE_INFO in enumerate(self.mp_list):
            print("Maximum Match Point : {}등".format(3-index))
            print("Image File : {}\nTest Image File : {}".format(SIFT_IMAGE_INFO[1], SIFT_IMAGE_INFO[2]))
            print("How many Image Feature point?")
            print("First Image SIFT keypoints = {}, Second Image SIFT keypoints = {}"
                  .format(len(SIFT_IMAGE_INFO[4]), len(SIFT_IMAGE_INFO[5])))
            print("Match Point={}".format(SIFT_IMAGE_INFO[3]))
        print("Finish!!")

    def save_MAXMP_image(self):
        print("Image File : {}\nTest Image File : {}".format(self.mp_list[-1][1], self.mp_list[-1][2]))
        MAX_MP_IMAGE_FILENAME = "MAX_MATCHING_" + self.mp_list[-1][1]
        print("Upload Maximum-Matching-Point Image:", MAX_MP_IMAGE_FILENAME)
        cv2.imwrite(MAX_MP_IMAGE_FILENAME, self.mp_list[-1][0], [cv2.IMWRITE_PNG_COMPRESSION])
        print("Finish!!")

    def save_progress(self, progress=0):
        self.progress = progress

    def return_progress(self):
        return self.progress


# Main Function
def main():
    # Initialize list
    image_list = []
    image_name = []
    test_image_list = []
    test_image_name = []
    # Clear Python Folder
    [os.remove(f) for f in glob.glob("/Users/kwondongjae/PycharmProjects/HW5_SIFT/*.png")]
    print("Clear Image Files")
    # Open Road Image
    progress = 0
    print("Start Open & Download Image")
    print_progressing(progress)
    for imgNum in images:
        progress += 1 / len(images) * 100
        imgNum_name = rename_FILENAME(imgNum)
        image = cv2.imread(imgNum, cv2.IMREAD_GRAYSCALE)
        image_name.append(imgNum_name)
        image_list.append(image)
        print_progressing(progress)
    stdout.write("\n")
    print("End Downloading")
    # Open Query Image
    progress = 0
    print("Start Open & Download Test Image")
    print_progressing(progress)
    for queryNum in querys:
        progress += 1 / len(querys) * 100
        queryNum_name = rename_FILENAME(queryNum)
        test = cv2.imread(queryNum, cv2.IMREAD_GRAYSCALE)
        test_image_name.append(queryNum_name)
        test_image_list.append(test)
        print_progressing(progress)
    stdout.write("\n")
    print("End Downloading")
    # Run SIFT
    progress = 0
    pw = 100 / len(test_image_list)
    SIFT = SIFT_Algorithms(pw=pw, IMAGE_list=image_list, IMAGE_FILENAME_list=image_name)
    print("Start Matching Image")
    for index, test_image in enumerate(test_image_list):
        SIFT.save_progress(progress=progress)
        SIFT.run_SIFT_function(test_image, filename=test_image_name[index])
        progress = SIFT.return_progress()
    stdout.write("\n")
    print("End Matching Image")
    # Upload Maximum Keypoint Image
    SIFT.save_MAX3KP_image()
    # Print Top 10 Maximum Matching-point Image
    SIFT.print_MAX3MP_INFO()
    # Save Maximum Matching-point Image
    SIFT.save_MAXMP_image()


# Run Main
if __name__ == '__main__':
    main()
