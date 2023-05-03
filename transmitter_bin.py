import argparse

from transmitter import Transmitter

parser = argparse.ArgumentParser(description='Transmits video through [tcpclientsink]. Performs object detection, tracking and visualization')

parser.add_argument('--host', help='Host address of [tcpclientsink]', required=True)
parser.add_argument('--port', help='Port number of [tcpclientsink]', required=True)
parser.add_argument('--weights_file', help='File containing weights to use for object detection', required=True)

parser.add_argument('--width', help='Width of received video frames', default=1632)
parser.add_argument('--height', help='Height of received video frames', default=924)
parser.add_argument('--enable_image_capture', help='Whether to enable image capture by pressing S', default=False)
parser.add_argument('--image_capture_dir', help='Path of directory to save captured images', default='./captures')
parser.add_argument('--confidence_threshold', help='Minimum confidence needed to detect object (0.0 to 1.0)', default=0.5)
parser.add_argument('--max_frames_disappeared', help='Maximum number of frames for object to have disappeared before unregistering', default=50)

args = vars(parser.parse_args())

def main():
    transmitter = Transmitter(
        host=args['host'],
        port=args['port'],
        width=args['width'],
        height=args['height'],
        weights_file=args['weights_file'],
        enable_image_capture=args['enable_image_capture'],
        image_capture_dir=args['image_capture_dir'],
        confidence_threshold=args['confidence_threshold'],
        max_frames_disappeared=args['max_frames_disappeared'],
    )

    transmitter.start()

    return

if __name__ == '__main__':
    main()