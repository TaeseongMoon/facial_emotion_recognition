import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
import tensorflow as tf
from os.path import join, dirname
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', '../data/train/emotion.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output', 'path to output image')
flags.DEFINE_integer('num_classes', 5, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    # yolo = YoloV3(classes=FLAGS.num_classes)
    # yolo = yolo.load_model(FLAGS.weights,custom_objects={'yolo_loss': YoloLoss})
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    get_output = 0
    for test_img in os.listdir(FLAGS.image):
        img_raw = tf.image.decode_image(
            open(join(FLAGS.image, test_img), 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            get_output += 1
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(join(FLAGS.output, test_img), img)
        logging.info('output saved to: {}'.format(FLAGS.output))
    print('output 개수 ', get_output)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass