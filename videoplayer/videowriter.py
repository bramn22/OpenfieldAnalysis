
import cv2

class VideoWriter:

    def __init__(self, output_path):
        self.output_path = output_path
        self.width, self.height, self.output, self.video, self.fps = None, None, None, None, None
        self.frames = []

    def add_trial(self, dlc, video_path, start, stop):
        df = dlc.df
        bodyparts = dlc.bodyparts

        def visualize_bodyparts(n, frame, prob_threshold=0.90):
            for key in bodyparts:
                x, y = int(df[key, 'x'][n]), int(df[key, 'y'][n])
                prob = df[key, 'likelihood'][n]
                if prob > prob_threshold:
                    cv2.circle(frame, (x, y), 5, color=(255, 0, 0), thickness=2)
                else:
                    cv2.circle(frame, (x, y), 5, color=(0, 0, 255), thickness=2)

            return frame

        cap = cv2.VideoCapture(video_path)
        print(start, stop)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        n = start
        while cap.isOpened():
            if n >= stop:
                break
            ret, frame = cap.read()
            if ret:

                res = visualize_bodyparts(n, frame)
                self.frames.append(res)

            else:
                break
            n += 1
        # cap.release()

    def write(self):
        self.width = self.frames[0].shape[1]
        self.height = self.frames[0].shape[0]
        self.video = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'XVID'), 30,
                                         (int(self.width), int(self.height)))

        for frame in self.frames:
            self.video.write(frame)
        self.video.release()
        cv2.destroyAllWindows()