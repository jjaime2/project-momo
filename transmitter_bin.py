import argparse

from transmitter import Transmitter

parser = argparse.ArgumentParser(description='Transmits video through [tcpclientsink]. Performs object detection, tracking and visualization')

parser.add_argument('--host', help='Host address of [tcpclientsink]', required=True)
parser.add_argument('--port', help='Port number of [tcpclientsink]', required=True)
parser.add_argument('--weights_file', help='File containing weights to use for object detection', required=True)

parser.add_argument('--width', help='Width of transmitted video frames', default=1920)
parser.add_argument('--height', help='Height of transmitted video frames', default=1080)
parser.add_argument('--frame_rate', help='Frame rate of transmitted video frames', default=30)
parser.add_argument('--confidence_threshold', help='Minimum confidence needed to detect object (0.0 to 1.0)', default=0.5)
parser.add_argument('--max_frames_disappeared', help='Maximum number of frames for object to have disappeared before unregistering', default=50)

args = vars(parser.parse_args())

def main():
    transmitter = Transmitter(
        host=args['host'],
        port=args['port'],
        width=args['width'],
        height=args['height'],
        frame_rate=args['frame_rate'],
        weights_file=args['weights_file'],
        confidence_threshold=args['confidence_threshold'],
        max_frames_disappeared=args['max_frames_disappeared'],
    )

    transmitter.start()

    return

if __name__ == '__main__':
    main()