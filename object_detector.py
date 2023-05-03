import torch


class ObjectDetector:

    def __init__(self, weights_file, confidence_threshold=0.5):

        self.weights_file = weights_file
        self.confidence_threshold = confidence_threshold
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', '/home/regulus/yolov5/runs/train/yolo_momo_detection_v2/weights/best.pt')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', self.weights_file)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, coords
    
    def class_to_label(self, class_id):

        return self.classes[int(class_id)]
