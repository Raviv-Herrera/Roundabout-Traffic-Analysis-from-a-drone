from video_class import DroneVideo


if __name__ == '__main__':

    v = DroneVideo(video_name='P1_roundabout.mp4')
    v.stabilize_video()
    v.generate_final_results()
