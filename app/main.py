import time
import datetime
import argparse


def main(args):
    if args.debug:
        print('Running in a debug mode')

    while True:
        print('Hello World')
        time.sleep(args.speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', dest='debug', action='store_true', help='A flag to run in debug mode')
    parser.add_argument('-speed', dest='speed', action='store', default=5, help='Running speed in seconds')

    main(parser.parse_args())
