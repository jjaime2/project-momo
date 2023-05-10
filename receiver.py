import logging
import os
import uuid

import cv2 as cv
import pathlib3x as pathlib
import object_detector as od
import object_tracker as ot


class Receiver():

    def __init__(
            self,
            host,
            port,
            weights_file,
            width,
            height,
            enable_image_capture,
            image_capture_dir,
            confidence_threshold,
            max_frames_disappeared,):

        self.host = host
        self.port = port
        self.weights_file = weights_file

        self.width = width
        self.height = height
        self.enable_image_capture = enable_image_capture
        self.image_capture_dir = pathlib.PurePath(image_capture_dir)
        self.confidence_threshold = float(confidence_threshold)
        self.max_frames_disappeared = max_frames_disappeared

        self.gstreamer_in = f'tcpclientsrc host={self.host} port={self.port} ! gdpdepay ! rtph264depay ! avdec_h264 ! videoscale ! video/x-raw, width=(int){self.width}, height=(int){self.height} ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink'
        self.capture = None
        self.detector = None
        self.tracker = None

    def start(self):

        detector = od.ObjectDetector(
            weights_file=self.weights_file,
            confidence_threshold=self.confidence_threshold,
        )
        tracker = ot.ObjectTracker(
            max_frames_disappeared=self.max_frames_disappeared
        )

        self.capture = cv.VideoCapture(self.gstreamer_in, cv.CAP_GSTREAMER)
        if not self.capture.isOpened():
            logging.error('Failed to open capture')
            return
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                logging.error('Failed to read from capture')
                break

            results = detector.score_frame(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            labels, coords = results
            num_labels = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]

            # Detect objects
            rects = []
            for label_index in range(num_labels):
                row = coords[label_index]
                confidence = row[4]

                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                    rects.append((x1, y1, x2, y2))
                    
                    cv.rectangle(
                        img=frame,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=(0, 255, 0),
                        thickness=2
                    )

                    cv.putText(
                        img=frame,
                        text=detector.class_to_label(labels[label_index]),
                        org=(x1, y1),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9,
                        color=(0, 255, 0),
                        thickness=2
                    )

            # Track objects
            objects = tracker.update(rects)
            for (object_id, centroid) in objects.items():
                cv.putText(
                    img=frame,
                    text=f"ID {object_id}",
                    org=(centroid[0] - 10, centroid[1] - 10),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2
                )

                cv.circle(
                    img=frame,
                    center=(centroid[0], centroid[1]),
                    radius=4,
                    color=(0, 255, 0)
                )

            cv.imshow('frame', frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info('Closing capture')
                break
            elif key == ord('s') and self.enable_image_capture:
                if not os.path.exists(self.image_capture_dir):
                    os.makedirs(self.image_capture_dir)

                frame_id = str(uuid.uuid4().hex[:8])
                frame_filename = pathlib.Path(self.image_capture_dir / frame_id).expanduser().append_suffix('.jpg')
                ret = cv.imwrite(str(frame_filename), frame)
                if not ret:
                    logging.error(f'Failed to write to {frame_filename}')
                    break
                logging.info(f'Saved frame to {frame_filename}')

        self.capture.release()

        return
