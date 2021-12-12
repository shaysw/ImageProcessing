import base64
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from sklearn import metrics
import os
import cv2

APP_NAME = "DigitRecognition"
APP_ID = 118
CLASSIFIER_FILE_PATH = "clf_4.p"
MNIST784_ZIP_FILE_PATH = "mnist_784.zip"


def load_mnist():
    print('Loading mnist784.zip...')
    all_data = pd.read_csv(MNIST784_ZIP_FILE_PATH, compression='zip')
    X = all_data.drop(columns='class')
    y = all_data['class']

    print('Split to train/test')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=10)

    return x_train, x_test, y_train, y_test


def load_classifier():
    if os.path.isfile(CLASSIFIER_FILE_PATH):
        with open(CLASSIFIER_FILE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return None


def fit_classifier():
    x_train, x_test, y_train, y_test = load_mnist()
    classifier = LinearSVC(random_state=0)
    print('Starting classifier fit...')
    start = time.perf_counter()
    classifier.fit(x_train, y_train)
    finish = time.perf_counter() - start
    print(f'Done. Fitting took - {finish} ms')
    y_predicted = classifier.predict(x_test)
    print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_predicted), "\n")

    with open(CLASSIFIER_FILE_PATH, 'wb+') as f:
        pickle.dump(classifier, f)

    return classifier


def invert(im):
    return 255 - im


def detect_edges(gray_image):  # Edge Detection, returns image array
    min_int, max_int, _, _ = cv2.minMaxLoc(gray_image)  # Grayscale: MinIntensity, Max, and locations
    beam = cv2.mean(gray_image)  # Find the mean intensity in the img pls.
    mean = float(beam[0])
    canny_of_tuna = cv2.Canny(gray_image, (mean + min_int) / 2,
                              (mean + max_int) / 2)  # Finds edges using thresholding and the Canny Edge process.
    return canny_of_tuna


def perform_digit_recognition(image_path):
    recognized_digits = []

    classifier = load_classifier()
    if not classifier:
        classifier = fit_classifier()

    # Read the input image
    im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    try:
        # make mask of where the transparent bits are
        trans_mask = im[:, :, 3] == 0

        # replace areas of transparency with white and not transparent
        im[trans_mask] = [255, 255, 255, 255]
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    except:
        pass

    im = invert(im)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    _, image_threshold = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours in the image
    contours, _ = cv2.findContours(detect_edges(image_threshold.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rectangles = [cv2.boundingRect(contour) for contour in contours]

    # For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
    for i, rect in enumerate(rectangles):
        try:
            # Draw the rectangles
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            center_x = x + int(w / 2)
            center_y = y + int(h / 2)
            length = max(w, h)
            if length > 110:
                length = length * 1.97
            else:
                length = length * 1.5
            # TODO: add empty pixels if out of bounds
            start_x = max(0, center_x - int(length / 2))
            finish_x = min(center_x + int(length / 2), len(image_threshold[0]))
            start_y = max(0, center_y - int(length / 2))
            finish_y = min(center_y + int(length / 2), len(image_threshold))
            roi = image_threshold[start_y:finish_y, start_x:finish_x]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imwrite(f'roi_{i}.png', roi)
            roi = roi.reshape(1, -1)

            nbr = classifier.predict(roi)
            recognized_digits.append(int(nbr[0]))
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        except Exception as e:
            recognized_digits.append(e)

    cv2.imwrite('test.png', im)
    return str(recognized_digits)


def upload_base64(base64_string):
    with open("imageToSave.png", "wb") as fh:
        a = base64.b64decode(base64_string)
        fh.write(a)


if __name__ == "__main__":
    # print(perform_digit_recognition(r"C:\Users\Shaysw\Documents\6_0.png"))
    print(perform_digit_recognition(r"C:\Users\Shaysw\Desktop\7.png"))
    # print(perform_digit_recognition("1 2 3.png"))
    # print(perform_digit_recognition("imageToSave.png"))
    # print(perform_digit_recognition(r"C:\Users\Shaysw\Documents\digit_recognition_image.png"))
