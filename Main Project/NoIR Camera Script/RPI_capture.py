import cv2
import picamera2

from picamera2 import Picamera2

picam2 = Picamera2()
# setup the video configurations
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888",
                                                         "size": (340,240)}))
picam2.set_controls({"FrameRate": 30})
picam2.start()

video_counter = 3
recording = False
video_writer = None

# image
image_counter = 543

# main exectuion loop
while True:
    frame = picam2.capture_array()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # applying equalised histogram for better contrast
    equalised = cv2.equalizeHist(gray)

    # show the real-time frame
    cv2.imshow("NoIR Camera Feed", equalised)

    #quick check if the recording and video are empty
    if recording and video_writer is not None:
        video_writer.write(cv2.cvtColor(equalised, cv2.COLOR_GRAY2BGR))

    # capturing keyboard keys
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'): #saves the file image
        filename = f"/home/pi/Pictures/Photos/carotid_artery_noir_{image_counter}.jpg"
        cv2.imwrite(filename, gray)
        print(f'Image saved to: {filename}')
        image_counter +=1

    if key == ord('v') and  not recording: #start recording the video
        video_filename = f"/home/pi/Pictures/Videos/carotid_artery_noir_{video_counter}.avi"
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (340,240))
        recording = True
        print(f'Recording started: {video_filename}')

    if key == ord('x') and recording: # stop the recording
        recording = False
        video_writer.release()
        video_writer = None
        print(f'Recording stopped at: {video_filename}')
        video_counter +=1

    #close the system
    if key == ord('q'):
        if recording:
            recording = False
            video_writer.release()
        break

cv2.destroyAllWindows()
picam2.close()