import time
import datetime
import argparse
import multiprocessing

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


def open_video_capture(input):
    capture = cv2.VideoCapture(input)

    if not capture.isOpened():
        raise RuntimeError(f'could not open input {input}')

    return capture


def open_load_model(model_path):
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def worker_main(args):
    print(f'opening input {args.input}', flush=True)
    capture = open_video_capture(args.input)

    # bind read_image function to capture
    def read_image():
        while True:
            _, img = capture.read()
            if img is not None:
                return img
            time.sleep(0.1)

    # inference
    def inferencing(interpreter, input_index, output_index, input_data):
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)

    print(f'loading model and model config', flush=True)
    interpreter = open_load_model(args.model)
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']

    while True:
        heartbeat.put(1)

        if args.verbose:
            print('reading image...', flush=True)
        image = read_image()
        if args.verbose:
            print('read image', flush=True)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(image, axis=0)

        if args.verbose:
            start_t = time.time()
        inferencing(interpreter, input_index, output_index, input_data)
        if args.verbose:
            print('{:.2f} elapsed for inferencing'.format(time.time - start_t))

        if args.interval != 0:
            if args.verbose:
                print('sleeping for {} seconds'.format(args.interval))
            time.sleep(args.interval)


def main(args):
    if args.verbose:
        print('running in a verbose mode', flush=True)

    heartbeat = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=worker_main, args=(args, heartbeat))

    try:
        worker.start()
        while True:
            heartbeat.get(timeout=30)  # throws an exception on timeout
    except Exception:
        pass

    # if we reach this point, the worker process has stopped
    worker.terminate()
    raise RuntimeError('worker is no longer responding')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-verbose', dest='verbose',
        action='store_true', help='Verbose')
    parser.add_argument(
        '-input', dest='input',
        action='store', default='/dev/video0',
        help='Path to input device or stream')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='mask_classifier_mobilenetv2_224_tuned.tflite',
        help='Path to model')
    parser.add_argument(
        '-image-size', dest='image_size',
        action='store', default=224,
        help='Input size of the model')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0,
        help='Time interval in seconds')

    main(parser.parse_args())
