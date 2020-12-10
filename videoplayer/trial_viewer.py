import cv2

def play_DLC_video_trials(dlc, video_path, trial_times):
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
    for (start, stop) in trial_times:
        print(start, stop)
        start = start - 30*1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        n = start
        stop = start + 30*6
        while cap.isOpened():
            if n >= stop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                n = start
            ret, frame = cap.read()
            if ret:
                res = visualize_bodyparts(n, frame)
                cv2.imshow('frame', res)
                key = cv2.waitKey(30)
                if key == ord("q"):
                    break
            else:
                break
            n += 1
    cap.release()
    cv2.destroyAllWindows()


def play_DLC_video_segment(dlc, video_path, start, stop):
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            n = start
        ret, frame = cap.read()
        if ret:
            res = visualize_bodyparts(n, frame)
            cv2.imshow('frame', res)
            key = cv2.waitKey(30)
            if key == ord("q"):
                break
        else:
            break
        n += 1
    cap.release()
    cv2.destroyAllWindows()