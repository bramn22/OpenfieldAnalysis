import cv2

class DataVideo:

    def __init__(self, file_path):
        self.file_path = file_path
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def play_video(self, fn=None, skip=0, **fn_args):
        cap = cv2.VideoCapture(self.file_path)
        n = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if n >= skip:
                    if fn is not None:
                        res = fn(n, frame, **fn_args)
                    else:
                        res = frame
                    cv2.imshow('frame', res)
                    key = cv2.waitKey(0)
                    if key == ord("q"):
                        break
            else:
                break
            n += 1
        cap.release()
        cv2.destroyAllWindows()

    def write_video(self, output_path, fn=None, skip=0, **fn_args):
        cap = cv2.VideoCapture(self.file_path)
        n = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height, output, video = None, None, None, None
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret:
                    print(n)
                    if n >= skip: # and n < len(labels):
                        if fn is not None:
                            res = fn(n, frame, **fn_args)
                        else:
                            res = frame
                        if width is None:
                            width = res.shape[1]
                            height = res.shape[0]
                            video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (int(width), int(height)))

                        video.write(res)
                else:
                    break
            except Exception as exc:
                print(f"Unable to process frame {n}. Skipping ...")
            n += 1
        cap.release()
        video.release()
        cv2.destroyAllWindows()

    def get_frame(self, index):
        cap = cv2.VideoCapture(self.file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        return frame