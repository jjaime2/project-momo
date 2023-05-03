import logging
import uuid

import cv2 as cv
import pathlib3x as pathlib


class Receiver():

    def __init__(
            self,
            host,
            port,
            width,
            height,
            enable_image_capture,
            image_capture_dir,):

        self.host = host
        self.port = port

        self.width = width
        self.height = height
        self.enable_image_capture = enable_image_capture
        self.image_capture_dir = pathlib.PurePath(image_capture_dir)

        self.gstreamer_in = f'tcpclientsrc host={self.host} port={self.port} ! gdpdepay ! rtph264depay ! avdec_h264 ! videoscale ! video/x-raw, width=(int){self.width}, height=(int){self.height} ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink'
        self.capture = None

    def start(self):

        self.capture = cv.VideoCapture(self.gstreamer_in, cv.CAP_GSTREAMER)
        if not self.capture.isOpened():
            logging.error('Failed to open capture')
            return
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                logging.error('Failed to read from capture')
                break

            cv.imshow('frame', frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info('Closing capture')
                break
            elif key == ord('s') and self.enable_image_capture:
                frame_id = str(uuid.uuid4().hex[:8])
                frame_filename = pathlib.Path(self.image_capture_dir / frame_id).expanduser().append_suffix('.jpg')
                ret = cv.imwrite(str(frame_filename), frame)
                if not ret:
                    logging.error(f'Failed to write to {frame_filename}')
                    break
                logging.info(f'Saved frame to {frame_filename}')

        self.capture.release()

        return
