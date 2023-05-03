import argparse

from receiver import Receiver

parser = argparse.ArgumentParser(description='Receives video from [tcpclientsrc]. Optionally performs object detection, tracking and visualization')

parser.add_argument('--host', help='Host address of [tcpclientsrc]', required=True)
parser.add_argument('--port', help='Port number of [tcpclientsrc]', required=True)

parser.add_argument('--width', help='Width of received video frames', default=1632)
parser.add_argument('--height', help='Height of received video frames', default=924)
parser.add_argument('--enable_image_capture', help='Whether to enable image capture by pressing S', default=False)
parser.add_argument('--image_capture_dir', help='Path of directory to save captured images', default='./captures')

args = vars(parser.parse_args())

def main():
    receiver = Receiver(
        host=args['host'],
        port=args['port'],
        width=args['width'],
        height=args['height'],
        enable_image_capture=args['enable_image_capture'],
        image_capture_dir=args['image_capture_dir'],
    )

    receiver.start()

    return

if __name__ == '__main__':
    main()