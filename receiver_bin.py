import argparse

from receiver import Receiver

parser = argparse.ArgumentParser(description='Receives video from [tcpclientsrc]. Optionally performs object detection, tracking and visualization')

parser.add_argument('--host', help='Host address of [tcpclientsrc]', required=True)
parser.add_argument('--port', help='Port number of [tcpclientsrc]', required=True)
parser.add_argument('--weights_file', help='File containing weights to use for object detection', required=True)

parser.add_argument('--width', help='Width of received video frames', default=1920)
parser.add_argument('--height', help='Height of received video frames', default=1080)
parser.add_argument('--enable_image_capture', help='Whether to enable image capture by pressing S', default=False)
parser.add_argument('--image_capture_dir', help='Path of directory to save captured images', default='./captures')
parser.add_argument('--confidence_threshold', help='Minimum confidence needed to detect object (0.0 to 1.0)', default=0.5)
parser.add_argument('--max_frames_disappeared', help='Maximum number of frames for object to have disappeared before unregistering', default=50)

args = vars(parser.parse_args())

def main():
    receiver = Receiver(
        host=args['host'],
        port=args['port'],
        weights_file=args['weights_file'],
        width=args['width'],
        height=args['height'],
        enable_image_capture=args['enable_image_capture'],
        image_capture_dir=args['image_capture_dir'],
        confidence_threshold=args['confidence_threshold'],
        max_frames_disappeared=args['max_frames_disappeared'],
    )

    receiver.start()

    return

if __name__ == '__main__':
    main()