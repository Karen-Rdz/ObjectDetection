from email import header
import os
from statistics import mean
from turtle import distance
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):

    header = ['frame', 'id', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'speed', 'energy', 'power', 'co2']
    with open('objects.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow(header)

    # Areas of interest to detect velocities
    # Test Gsada 1
    area_1 = [(0, 400), (500, 400), (500, 600), (0, 600)]
    area_2 = [(0, 600), (500, 600), (500, 800), (0, 800)]
    area_3 = [(0, 800), (715, 800), (715, 1200), (0, 1200)]  

    # Test Gsada 2
    # area_1 = [(0, 200), (715, 200), (715, 400), (0, 400)]
    # area_2 = [(0, 700), (715, 700), (715, 1100), (0, 1100)]

    objects_entering = {}
    objects_elapsed_time = {}
    all_speeds = []
    all_power = []
    all_c02 = []
    all_energy = []
    masa = 1200

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            final_data = ['Average speed: ' + str(mean(all_speeds)), 
                        'Average energy: ' + str(mean(all_energy)), 
                        'Total energy: ' + str(sum(all_energy)),
                        'Average power: ', str(mean(all_power)), 
                        'Total power: ', str(sum(all_power)), 
                        'Average emmissions: ', str(mean(all_c02)), 
                        'Total emissions: ', str(sum(all_c02))]
            with open('final_results.txt', 'w') as f:
                        f.write('\n'.join(final_data))
            for value in final_data: print(value)
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1

        # For more speed:
        if frame_num % 2 != 0:
           continue

        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()   

        # obtain center of box
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            a_speed_km = 0 
            a_speed_km_2 = 0
            energia_c = 0
            emisiones_co2 = 0
            power = 0
            
        # exit area 1
            result_area = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(cx), int(cy)), False) 
            if result_area >= 0:
                objects_entering[track.track_id] = frame_num

        # enter area 2
            if track.track_id in objects_entering:
                result_area = cv2.pointPolygonTest(np.array(area_2, np.int32), (int(cx), int(cy)), False)
                if result_area >= 0:
                    # time_between = frame_num
                    time_between = frame_num - objects_entering[track.track_id]
                    # calculate average veocity
                    distance = 20 # meters
                    if time_between > 0:
                        a_speed_ms_1 = distance / (time_between/30)
                        a_speed_km_1 = a_speed_ms_1*3.6

        # enter area 3
            if track.track_id in objects_entering:
                result_area = cv2.pointPolygonTest(np.array(area_3, np.int32), (int(cx), int(cy)), False)
                if result_area >= 0:
                    # elapsed_time = time.time() - objects_entering[track.track_id]
                    elapsed_time = frame_num - objects_entering[track.track_id]
                    if track.track_id not in objects_elapsed_time:
                        objects_elapsed_time[track.track_id] = elapsed_time
                    if track.track_id in objects_elapsed_time:
                        elapsed_time = objects_elapsed_time[track.track_id]
                    # calculate average veocity
                    distance = 20 # meters
                    if elapsed_time > 0:
                        a_speed_ms = distance / (elapsed_time/30)
                        a_speed_km = a_speed_ms*3.6

                    if frame_num-time_between > 0:
                        a_speed_ms_2 = distance / ((frame_num-time_between)/30)
                        a_speed_km_2 = a_speed_ms_2*3.6
                        acceleration = (a_speed_km_2 - a_speed_km_1) / (frame_num - time_between)
                    else:
                        acceleration = 1

                    print(a_speed_km_2, a_speed_km_1)
                    print(frame_num - time_between)
                    print(masa, acceleration, a_speed_km)
                    power = masa*acceleration*a_speed_km
                    energia_c = (a_speed_km*a_speed_km*masa)/2
                    emisiones_co2 = (distance/1000)*251

                    all_speeds.append(a_speed_km)
                    if energia_c != 0: all_energy.append(energia_c)
                    if power != 0: all_power.append(power)
                    if emisiones_co2 != 0: all_c02.append(emisiones_co2)

                    # draw bbox on screen
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.putText(frame, class_name + "-" + str(track.track_id) + "-" + str(int(a_speed_km)) + "km/h",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw areas of interes
            for area in [area_1, area_2, area_3]:
                cv2.polylines(frame, [np.array(area, np.int32)], True, (15, 200, 10), 6)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            data = [frame_num, str(track.track_id), class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), a_speed_km, energia_c, power, emisiones_co2]

            with open('objects.csv', 'a') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow(data)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
