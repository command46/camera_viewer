from flask import Flask, request, jsonify
import os
import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_photos'  # 照片保存的文件夹
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return jsonify({'message': 'No imageFile part'}), 400
    file = request.files['imageFile']
    camera_type = request.form.get('cameraType', 'unknown') # 获取摄像头类型，默认为 unknown

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{timestamp}-{camera_type}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'message': 'Upload failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345, debug=True) # 监听所有网络接口，方便电脑和手机在同一网络下访问