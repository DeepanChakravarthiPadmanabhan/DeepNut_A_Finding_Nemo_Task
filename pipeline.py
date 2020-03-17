import numpy as np
import cv2 as cv
import tensorflow as tf
import math
import argparse
import os
import pandas as pd
import sys
import time
import datetime

label_map = {1:'Peanut',2:'Walnut',3:'Hazelnut'}

TEST = False

def show_frame(image, caption='Frame to see'):
    cv.imshow(caption, image)
    cv.waitKey(0)

def save_img(image, filename):
    cv.imwrite(filename, image)


def capture_frames(video_data):
    captured_frames = list()
    img_dict = dict()
    img_list = list()
    img_idx = list()
    frame_diff_list = list()

    cap = cv.VideoCapture(video_data)
    n_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)


    count = 0
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        frame_time = time.time()
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(fps / 2.0) == 0):
            b, g, r = cv.split(frame)
            img = cv.merge((b, g, r))
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_dict[count] = (gray_image)
        count = count+1
    cap.release()
    if TEST:
        print("Frame: {}\nFrames per second: {}".format(n_frames, fps))

    k = img_dict.items()
    for n,i in enumerate(k):
        img_list.append(i[1])
        img_idx.append(i[0])

    # calculating the frame difference between consecutive frames
    for i in range(1, len(img_list)):
        frame_diff = cv.absdiff(img_list[i], img_list[i - 1])
        frame_diff = cv.GaussianBlur(frame_diff, (3, 3), 0)
        frame_diff = cv.threshold(frame_diff, 25, 255, cv.THRESH_BINARY)[1]
        frame_diff_list.append(cv.countNonZero(frame_diff))


    min_idx = np.argmin(frame_diff_list)
    if len(frame_diff_list) > 6 and min_idx > int(len(frame_diff_list) * 0.8):
        min_idx = np.argmin(frame_diff_list[:-2])

    # storing the frame with the least consecutive frame difference for further processing
    captured_frames.append(img_list[min_idx + 1])

    return captured_frames[0], img_idx[min_idx+1]

def run_inference(frozen_graph_path, stable_frame):

    # Read the graph.
    with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = stable_frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        class_label = list() # list of labelIDs
        bbox_list = list() # list of tuples - x,y, right, bottom

        for i in range(num_detections):
            classId = int(out[3][0][i])
            class_label.append(classId)
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                tuple_data = (x, y, right, bottom)
                bbox_list.append(tuple_data)
                if TEST:
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    if TEST:
        show_frame(img, 'TensorFlow MobileNet-SSD')

    return class_label, bbox_list


def run_inference_opencv(frozen_graph_path, stable_frame):
    cvNet = cv.dnn.readNetFromTensorflow(frozen_graph_path, 'graph.pbtxt')

    img = stable_frame
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    cv.imshow('img', img)
    cv.waitKey()

def get_point(bounding_box):

    x,y = bounding_box[0], bounding_box[1]
    right, bottom = bounding_box[2], bounding_box[3]
    center_top = (x + right)/2.
    center_right = (y + bottom)/2.

    return center_top, center_right

def get_label_marker_position(bounding_box):

    x, y = bounding_box[0], bounding_box[1]
    x_marker = x
    y_marker = y-5

    return x_marker, y_marker

def get_output_format(class_label, bounding_box, selected_frame):
    label_name = list()
    for i in class_label:
        label_name.append(label_map[int(i)])

    x_list = list()
    y_list = list()
    for i in bounding_box:
        center_x, center_y = get_point(i)
        x_list.append(center_x)
        y_list.append(center_y)

    lines_in_output = list()

    for i in zip(x_list, y_list, label_name):
        lines_in_output.append(str(selected_frame)+','+str(int(i[0]))+','+str(int(i[1]))+','+i[2])

    return(lines_in_output)

def enhance_CLAHE(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # show_frame(cl, 'CLAHE output')

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    return final

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepNut- A finding Nemo Task')
    parser.add_argument("inp", type=str, help="input video path")
    parser.add_argument("model", type=str, help="frozen model path")
    parser.add_argument("out", type=str,nargs='?', default='Result/', help="output csv")

    args = parser.parse_args()

    video_data = args.inp
    frozen_graph_path = args.model
    output_path = args.out
    output_file_name = (os.path.basename(args.inp)).split('.')[0]


    sys.stdout = open('Logger.txt', 'w')
    print("Input reading successful [%s]" % str(datetime.datetime.now()))

    frozen_graph_path = 'frozen_model/frozen_inference_graph.pb'
    start_time = time.time()
    stable_frame, selected_frame = capture_frames(video_data)

    print("Stable frame extraction completed [%s]" % str(datetime.datetime.now()))

    if TEST:
        print("Selected frame", selected_frame)

    cap = cv.VideoCapture(video_data)
    cap.set(1, selected_frame)
    ret, stable_frame = cap.read()

    if TEST:
        stable_frame = enhance_CLAHE(stable_frame)

    print("Running inference [%s]" % str(datetime.datetime.now()))

    class_label, bounding_box = run_inference_opencv(frozen_graph_path, stable_frame)

    print("Inference run completed [%s]" % str(datetime.datetime.now()))

    lines_in_output = get_output_format(class_label, bounding_box, selected_frame)

    if TEST:
        # To draw bounding box and label the predictions.

        x_list = list()
        y_list = list()
        label_color_list = list()
        label_name_list = list()
        label_color = {1:(255,0,0), 2:(0,255,0), 3:(0,0,255)}
        for i in bounding_box:
            center_x, center_y = get_point(i)
            x_list.append(center_x)
            y_list.append(center_y)

        x_marker = list()
        y_marker = list()
        for i in bounding_box:
            x_m, y_m = get_label_marker_position(i)
            x_marker.append(x_m)
            y_marker.append(y_m)

        for i in class_label:
            label_color_list.append(label_color[int(i)])

        for i in class_label:
            label_name_list.append(label_map[int(i)])

        for i in zip(x_list, y_list, label_color_list):
            cv.circle(stable_frame, (int(i[0]), int(i[1])), 2, i[2], 5)

        for i in zip(label_name_list, x_marker, y_marker, label_color_list):
            cv.putText(stable_frame,i[0], (int(i[1]),int(i[2])), cv.FONT_HERSHEY_SIMPLEX, 0.8, i[3], 2,cv.LINE_AA)
        
        # To show the output
        show_frame(stable_frame)

        save_img(stable_frame, 'Result_'+output_file_name+'.jpg')  # Args- image, filename

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Formatting output [%s]" % str(datetime.datetime.now()))

    file_name = output_path+'/'+output_file_name+'.csv'

    results = pd.DataFrame([lines_in_output])
    results.T.to_csv(file_name, encoding='utf-8', header=False, index=False)

    end_time = time.time()
    print("Time taken to extract stable frame:", end_time - start_time)
    print("I completed my task. Chau. [%s]" % str(datetime.datetime.now()))


