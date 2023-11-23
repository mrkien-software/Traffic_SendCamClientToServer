from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time

import tensorflow as tf
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model biển báo
model_path = "model_traffic.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Chỉ lấy biển báo màu đỏ màu xanh lam
def returnRedAndBlue(img):
	yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y,u,v=cv2.split(yuv)

    # chuyển ảnh biển báo giao thông sang màu red và blue để check
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	check_red = np.sum((rgb[:,:,0] > 150) & (rgb[:,:,1] < 100) & (rgb[:,:,2] < 100))
	check_blue = np.sum((rgb[:,:,0] < 100) & (rgb[:,:,1] < 100) & (rgb[:,:,2] > 150))
    # Tính tỷ lệ  màu nào nhiều hơn
	ratio = check_red / (check_blue + 1)  # Thêm 1 để tránh chia cho 0
    # Xác định màu chủ đạo
	if ratio >0.1:
		return v
	else:
		return u
	
# Chỉ lấy biển báo màu đỏ 
def returnRedness(img): 
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return v

# Áp dụng ngưỡng nhị phân cho hình ảnh.
def threshold(img,T=150):
	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def findContour(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
    if not contours:
        return None  # Return None if no contours are found
    c = [cv2.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x,y,w,h=cv2.boundingRect(contours)
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	sign=img[y:(y+h) , x:(x+w)]
	return img,sign

def boundaryBoxNew(img, contours):
    x, y, w, h = cv2.boundingRect(contours)
    img_vung_chon = img[y:y + h, x:x + w].copy()
    return img_vung_chon

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(frame_data):
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    cv2_im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    
    # Lưu lại ảnh vào folder
    current_time_ticks = int(time.time() * 1000)  # multiplying by 1000 to get milliseconds
    filename = ""
    filename = f"img/kien_{current_time_ticks}.jpg"
    cv2.imwrite(filename, img)
    
    # logic
    image_fromarray = Image.fromarray(img, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255
    pred = loaded_model.predict(input_data)
    result = pred.argmax()
    prediction = ""
    if result == 0:
        prediction = 'Giới hạn tốc độ (20km/h)'
    elif result == 1:
        prediction = 'Giới hạn tốc độ (30km/h)'
    elif result == 2:
        prediction = 'Giới hạn tốc độ (50km/h)'
    elif result == 3:
        prediction = 'Giới hạn tốc độ (60km/h)'
    elif result == 4:
        prediction = 'Giới hạn tốc độ (70km/h)'
    elif result == 5:
        prediction = 'Giới hạn tốc độ (80km/h)'
    elif result == 6:
        prediction = 'Hết giới hạn tốc độ (80km/h)'
    elif result == 7:
        prediction = 'Giới hạn tốc độ (100km/h)'
    elif result == 8:
        prediction = 'Giới hạn tốc độ (120km/h)'
    elif result == 9:
        prediction = 'Cấm vượt'
    elif result == 10:
        prediction = 'Cấm vượt xe trên 3,5 tấn'
    elif result == 11:
        prediction = 'Quyền ưu tiên tại ngã 4'
    elif result == 12:
        prediction = 'Đường ưu tiên'
    elif result == 13:
        prediction = 'Nhường đường xe hướng khác'
    elif result == 14:
        prediction = 'Dừng lại'
    elif result == 15:
        prediction = 'Cấm xe cộ'
    elif result == 16:
        prediction = 'Cấm xe > 3,5 tấn'
    elif result == 17:
        prediction = 'Cấm vào đường này'
    elif result == 18:
        prediction = 'Cảnh báo đường nguy hiểm'
    elif result == 19:
        prediction = 'Bên trái đường cong nguy hiểm'
    elif result == 20:
        prediction = 'Bên phải đường cong nguy hiểm'
    elif result == 21:
        prediction = 'Đường cong nguy hiểm'
    elif result == 22:
        prediction = 'Đường gập ghềnh'
    elif result == 23:
        prediction = 'Đường trơn'
    elif result == 24:
        prediction = 'Đường bị thu hẹp ở bên phải'
    elif result == 25:
        prediction = 'Đang làm đường'
    elif result == 26:
        prediction = 'Có Tín hiệu đèn'
    elif result == 27:
        prediction = 'Làn người đi bộ'
    elif result == 28:
        prediction = 'Có trẻ em băng qua'
    elif result == 29:
        prediction = 'Xe đạp băng qua đường'
    elif result == 30:
        prediction = 'Coi chừng băng/tuyết'
    elif result == 31:
        prediction = 'Động vật hoang dã băng qua'
    elif result == 32:
        prediction = 'Tốc độ kết thúc + giới hạn vượt qua'
    elif result == 33:
        prediction = 'Rẽ phải'
    elif result == 34:
        prediction = 'Rẽ trái'
    elif result == 35:
        prediction = 'Đi thẳng phía trước'
    elif result == 36:
        prediction = 'Đi thẳng hoặc rẽ phải'
    elif result == 37:
        prediction = 'Đi thẳng hoặc rẽ trái'
    elif result == 38:
        prediction = 'Đi bên phải'
    elif result == 39:
        prediction = 'Đi bên trái'
    elif result == 40:
        prediction = 'Bùng binh bắt buộc'
    elif result == 41:
        prediction = 'Đường cụt, không được vượt qua'
    elif result == 42:
        prediction = 'Đường cụt xe > 3,5 tấn không vượt qua được'
    else:
        prediction = ''

    if(result < 9):
        print(prediction)
        # Xử lý khung hình video nhận được từ máy khách
        print('Received video frame:', len(frame_data), 'bytes')
        # Bạn có thể phát khung tới tất cả các máy khách được kết nối nếu cần
        emit('video_frame', prediction, broadcast=True)

if __name__ == '__main__':
    # Use eventlet to run the application with SSL
    import eventlet
    import eventlet.wsgi
    from eventlet import wrap_ssl
    
    eventlet.wsgi.server(wrap_ssl(eventlet.listen(('0.0.0.0', 5000)), certfile='localhost.crt', keyfile='localhost.key', server_side=True),
                        app,
                        debug=False,
                        log_output=False)