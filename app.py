from flask import Flask, render_template, Response, jsonify, request
from test import generate_frames, detection_output, stop_detection
import geocoder

app = Flask(__name__, static_folder='static')

location = geocoder.ip('me')
coords = location.latlng if location.ok else ["Unavailable", "Unavailable"]

@app.route('/')
def index():
    return render_template('index.html', coords=coords)

@app.route('/detection_status')
def detection_status():
    return jsonify(detection_output)

@app.route('/start', methods=['POST', 'GET'])
def start_detection_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_detection_route():
    stop_detection()
    return "", 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
