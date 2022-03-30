from flask import Flask, render_template, Response
from camera import VideoCamera
import logging
import os

app = Flask(__name__)


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir="applicationLogs"
general_logs = "logs"
general_log_path_dir=os.path.join(log_dir,general_logs)

os.makedirs(general_log_path_dir, exist_ok=True)
general_logs_name = "general_logs.log"
general_log_path = os.path.join(general_log_path_dir,general_logs_name)
print(general_log_path)
logging.basicConfig(filename = general_log_path, level=logging.INFO, format=logging_str)
logging.info("Application started")

@app.route('/')
def index():
    logging.info("Rendering index.html")
    return render_template('index.html')

def gen(camera):
    while True:
        logging.info("Generating frames")
        frame = camera.get_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    logging.info("Rendering video feed")
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
