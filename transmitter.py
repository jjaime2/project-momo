import logging

import cv2 as cv
import pathlib3x as pathlib
import object_detector as od
import object_tracker as ot


class Transmitter():

    def __init__(
            self,
            host,
            port,
            width,
            height,
            frame_rate,
            weights_file,
            confidence_threshold,
            max_frames_disappeared,):

        self.host = host
        self.port = port
        self.weights_file = weights_file

        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.confidence_threshold = float(confidence_threshold)
        self.max_frames_disappeared = max_frames_disappeared

        self.gstreamer_in = f'nvarguscamerasrc ! video/x-raw(memory:NVMM), width={self.width}, height={self.height}, framerate={self.frame_rate}/1 ! nvvidconv flip-method=2 ! omxh264enc ! h264parse ! rtph264pay config-interval=1 pt=96 ! gdppay ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink'
        self.gstreamer_out = f'appsrc ! queue ! tcpserversink host={self.host} port={self.port}'
        self.capture = None
        self.writer = None
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
        
        self.writer = cv.VideoWriter(self.gstreamer_out, cv.CAP_GSTREAMER, 0, self.frame_rate, (self.width, self.height))
        if not self.writer.isOpened():
            logging.error('Failed to open writer')
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

            self.writer.write(frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info('Closing capture')
                break

        self.capture.release()
        self.writer.release()

        return
