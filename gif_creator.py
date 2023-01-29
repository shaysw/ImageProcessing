import cv2, imageio, io

APP_NAME = "GifCreator"
APP_ID = 119

def create_gif(uploaded_video_path):
    bytes_io = io.BytesIO()
    capture = cv2.VideoCapture(uploaded_video_path)
    frame_list = []

    while True:
        ret, frame = capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
            
    capture.release()
    cv2.destroyAllWindows()

    # Convert to gif using the imageio.mimsave method
    imageio.mimsave(bytes_io, frame_list, 'gif', fps=60)
    bytes_io.seek(0)
    
    return bytes_io