
import cv2


class VideoProcessor:

    def __init__(self, video_path):

        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError("Unable to open video file")

        # Extract video metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

    def display_video_info(self):

        print("Video Information")
        print("----------------------")

        print("FPS:", self.fps)
        print("Total Frames:", self.total_frames)

        print("Width:", self.width)
        print("Height:", self.height)

        print("Duration:", self.duration, "seconds")

    def read_frames(self):

        frame_index = 0

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            yield frame_index, frame

            frame_index += 1

    def reset(self):
        """Reset video capture pointer to frame 0."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):

        self.cap.release()
