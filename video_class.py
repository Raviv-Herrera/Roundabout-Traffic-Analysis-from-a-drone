import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from loguru import logger
from alive_progress import alive_bar
from utils import clean_duplicate, classify_color, count_num_of_vehicles, COLOR_LUT


class DroneVideo:

    def __init__(self, video_name: str):

        self._video_name = video_name

        self._feature_params = dict(maxCorners=10000,
                                    qualityLevel=0.01,
                                    minDistance=10,
                                    blockSize=5)
        self._lk_params = dict(winSize=(15, 15),
                               maxLevel=15,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         10, 0.03))

        self._original_video: cv2.VideoCapture = self.__load_video()
        self._stabilized_video: cv2.VideoCapture = self._original_video

    def __load_video(self) -> cv2.VideoCapture:
        """
        This method loads and inserts the video for the DroneVideo class
        :return:
        """
        return cv2.VideoCapture(self._video_name)

    @staticmethod
    def play_video(video: cv2.VideoCapture) -> None:
        """
        This method displays the inserted video of the DroneVideo class
        :param video: (cv2.VideoCapture) a video to display
        :return:
        """
        while True:

            ret, frame = video.read()  # read next frame

            if ret:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xff == 27:  # ESC key pressed?
                    break
            else:
                break

        video.release()  # release input video
        cv2.destroyAllWindows()  # delete output window
        cv2.waitKey(1)

    def cut_black_edges(self):

        _, frame0 = self._stabilized_video.read()
        frame0 = frame0[260:1270, 705:1750]
        frame0 = cv2.resize(frame0, (600, 600))

        h, w, _ = frame0.shape
        FPS = self._stabilized_video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        output_name = 'P1_roundabout_Stabilized_cut.avi'
        out = cv2.VideoWriter(output_name, fourcc, FPS, (w, h))

        while True:
            ret, frame1 = self._stabilized_video.read()  # read next frame

            if ret:
                frame1 = frame1[270:1270, 705:1750]
                frame1 = cv2.resize(frame1, (600, 600))
                # cv2.imshow('WINDOW_NAME', frame1)
                out.write(frame1)
                if cv2.waitKey(1) & 0xff == 27:  # ESC key pressed?
                    break
            else:
                break

        self._stabilized_video = cv2.VideoCapture(os.path.join(os.getcwd(), f"{output_name}"))
        out.release()  # release output video
        cv2.destroyAllWindows()  # delete output window
        cv2.waitKey(1)

    def stabilize_video(self, N: int = 250) -> None:
        """

        :param N:
        :return:
        """
        logger.info("Starts stabilizing the Video ... ")
        _, frame_prev = self._original_video.read()

        frame_prev = cv2.copyMakeBorder(frame_prev, N, N, N, N, cv2.BORDER_CONSTANT)
        frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

        # generate a mask for the roundabout's middle as an 'anchor points'
        mask_of_roundabout = np.zeros_like(frame_prev_gray)
        mask_of_roundabout[400:1150, 900:1590] = 255

        pts_prev = cv2.goodFeaturesToTrack(frame_prev_gray,
                                           mask=mask_of_roundabout,
                                           **self._feature_params)
        pts_ref = pts_prev.copy()
        h, w = frame_prev_gray.shape

        FPS = self._original_video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        output_name = 'P1_roundabout_Stabilized.avi'
        out = cv2.VideoWriter(output_name, fourcc, FPS, (w, h))

        while True:
            ret, frame_next = self._original_video.read()
            if ret:

                frame_next = cv2.copyMakeBorder(frame_next, N, N, N, N, cv2.BORDER_CONSTANT)
                frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)

                # the optical flow is being done among the previous frame's good points and the next frame's good points
                pts_next, status, err = cv2.calcOpticalFlowPyrLK(frame_prev_gray,
                                                                 frame_next_gray,
                                                                 pts_prev,
                                                                 None,
                                                                 **self._lk_params)

                pts_next = pts_next[status[:, 0] == 1]
                pts_ref = pts_ref[status[:, 0] == 1]

                # calculate homography among the anchor points and the next points
                H, _ = cv2.findHomography(pts_next, pts_ref, cv2.RANSAC, 22.0)
                frame_next_warp = cv2.warpPerspective(frame_next, H, (w, h))

                frame_prev_gray = frame_next_gray
                pts_prev = pts_next
                out.write(frame_next_warp)

                if cv2.waitKey(1) & 0xff == 27:
                    break

            else:
                break

        self._stabilized_video = cv2.VideoCapture(os.path.join(os.getcwd(), f"{output_name}"))
        self.cut_black_edges()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logger.success("finished stabilizing the Video ... ")

    def display_stab(self):

        cap = self._stabilized_video

        while True:
            ret, frame1 = cap.read()  # read next frame

            if ret:
                cv2.imshow('WINDOW_NAME', frame1)
                if cv2.waitKey(1) & 0xff == 27:  # ESC key pressed?
                    break
            else:
                break

        cap.release()  # release input video
        cv2.destroyAllWindows()  # delete output window
        cv2.waitKey(1)

    def generate_final_results(self) -> None:
        """

        :return:
        """
        self._feature_params = dict(maxCorners=15,
                                    qualityLevel=0.2,
                                    minDistance=60,
                                    blockSize=2)

        self._lk_params = dict(winSize=(15, 15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS
                                                                       | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        ret, F0 = self._stabilized_video.read()  # read first frame
        F0_gray = cv2.cvtColor(F0, cv2.COLOR_BGR2GRAY)  # convert the first frame to gray scale

        # Creating a mask for looking for good points to track only in the roads

        mask = np.zeros_like(F0_gray)
        mask = cv2.circle(mask, (300, 290), 290, thickness=-1, color=255)
        mask = cv2.circle(mask, (300, 300), 160, thickness=-1, color=0)

        pts0 = cv2.goodFeaturesToTrack(F0_gray, mask=mask, **self._feature_params)  # looking for a good points to track
        canvas = np.zeros_like(F0)
        canvas1 = np.zeros_like(F0)
        counter = 0

        while True:

            ret, F1 = self._stabilized_video.read()  # read next frame

            if ret:

                F1_gray = cv2.cvtColor(F1, cv2.COLOR_BGR2GRAY)  # convert to gray scale

                pts1, status, err = cv2.calcOpticalFlowPyrLK(F0_gray, F1_gray, pts0, None,
                                                             **self._lk_params)  # calculate Optical Flow

                # Delete untracking points
                pts1 = pts1[status[:, 0] == 1]
                pts0 = pts0[status[:, 0] == 1]

                std = 2  # define a a standart diviation

                new = []
                # checking which point is actually moving , if it is moving so append it to 'new' list

                for i in range(pts0.shape[0]):
                    if (np.abs(pts0[i][0][0] - pts1[i][0][0]) > std) or (np.abs(pts0[i][0][1] - pts1[i][0][1]) > std):
                        new.append([pts1[i][0][0], pts1[i][0][1]])

                new = np.asarray(new)

                # draw the moving points
                for i in range(new.shape[0]):

                    cv2.circle(canvas1, (int(new[i][0]), int(new[i][1])), 4, (255, 255, 0), -1)
                    cv2.circle(canvas, (int(new[i][0]), int(new[i][1])), 2, (255, 255, 255), -1)

                    cv2.rectangle(F1, (int(new[i][0]) - 25, int(new[i][1]) - 25),
                                  (20 + int(new[i][0]), 20 + int(new[i][1])), (255, 50, 50), 2)
                    if 600 > int(new[i][0]) > 30 and 30 < int(new[i][1]) < 600:
                        color222 = classify_color(int(new[i][0]), int(new[i][1]), F1, (18, 6))

                    # ========== Option to show the final result with color identifier
                    #             cv2.putText(F1, f"Vehicle {color222}",(20+int(new[i][0]), 20+int(new[i][1])),
                    #              cv2.FONT_HERSHEY_PLAIN,1, (0,0,250), 2 )
                    #           # ==========Option to show the final result without color identifier
                    cv2.putText(F1, f"Vehicle ", (20 + int(new[i][0]), 20 + int(new[i][1])), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 250), 2)

                pts0 = new
                pts0 = np.reshape(pts0, (pts0.shape[0], 1, pts0.shape[1]))  # reshape pts0 into appropiate format

                # Generate a heatmap
                # bluring canvas and apply a color map

                blur = cv2.GaussianBlur(canvas, (0, 0), 6)

                heatmap = cv2.applyColorMap(blur, cv2.COLORMAP_HOT)

                canvas1 = np.uint8(canvas1 * 0.8)  # fade out canvas1
                final = cv2.add(F1, canvas1)

                # add final and heatmap to show it later
                w = cv2.addWeighted(final, 1, heatmap, 0.55, 0)

                counter += 1

                # every 25 frames we look for new points to track
                if counter % 25 == 0:
                    new_points = cv2.goodFeaturesToTrack(F1_gray, mask=mask, **self._feature_params)
                    r = clean_duplicate(new_points, pts0)

                    # add new_points to pts0

                    new_points = new_points[r]
                    pts0 = np.vstack((pts0, new_points))

                cv2.putText(w, f"Vehicle's Number :{count_num_of_vehicles(pts0, mask)}", (110, 300), 3, 1, (0, 255, 255))

                # Show results
                # ================= Videos of the final result =======================================
                # =========Video w ->> The Final result =========================
                cv2.imshow('w', w)
                cv2.imshow('blur', blur)
                # =========Video heatmap -->> shows only heatmap =======================================
                cv2.imshow('heatmap', heatmap)
                # ========= Video F1 -->> shows the Video with labels ==================================
                cv2.imshow('F1', F1)
                # =========== Video canvas -->> shows the pre-stage before heatmap ======================
                cv2.imshow('canvas', canvas)

                F0_gray = F1_gray

                if cv2.waitKey(10) & 0xff == 27:  # ESC key pressed?
                    break
            else:
                break

        self._stabilized_video.release()  # release input video
        cv2.destroyAllWindows()  # delete output window
        cv2.waitKey(1)
