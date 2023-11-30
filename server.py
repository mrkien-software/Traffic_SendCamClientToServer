from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import time
import base64

import tensorflow as tf
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model biển báo
model_path = "Traffic_30_30.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Check xem có phải biển báo hình tròn, tam giác, vuông không
# Độ chính xác của hình ảnh trên 60% chấp nhận đúng
def is_circle_or_triangle(img):
    # Tìm đường viền trong ảnh đen trắng
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kiểm tra xem có phát hiện được đường viền nào không
    if contours:
        # Giả sử chỉ có đường viền lớn nhất là phù hợp
        largest_contour = max(contours, key=cv2.contourArea)

        # Xấp xỉ đường viền cho một đa giác
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # hình có 3,4,5 cạnh cũng ghi nhận
        if 2 < len(approx) < 6:
            return 1
        
        # Check if the contour is approximately a circle
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > 0.5: #giống tầm 50% hình tròn
            return 1
    return 0
    
def compare_one_image(img_goc, img_check):
    # cho size img_check = img_goc
    img_goc = cv2.resize(img_goc, (30,30))
    img_check = cv2.resize(img_check, (30,30))

    # Convert images to grayscale
    img_goc_gray = cv2.cvtColor(img_goc, cv2.COLOR_BGR2GRAY)
    img_check_gray = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)

    ssi_index, _ = ssim(img_goc_gray, img_check_gray, full=True)
    return ssi_index

def compare_in_folder(img_check, archive_folder):
    if len(os.listdir(archive_folder)) == 0:
        # Folder trống insert cái đầu tiên
        file_name = 'img/dau_tien.jpg'
        cv2.imwrite(file_name, img_check)
    else:  
        for filename in os.listdir(archive_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
                img_goc_in_store = cv2.imread(os.path.join(archive_folder, filename))
                ssi_index = compare_one_image(img_goc_in_store, img_check)
                
                if 0.1 < ssi_index < 0.70: # giống tầm 70% trở đi thì coi như đã có trong kho ko insert thêm nữa
                    per = ssi_index*100
                    file_name = 'img/giong_' + str(per) + '.jpg'
                    cv2.imwrite(file_name, img_check)

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
    
# Làm mờ hình ảnh để giảm nhiễu và cải thiện khả năng phát hiện
def filteringImages(img):
    img=cv2.GaussianBlur(img,(11,11),0)
    return img

# Chỉ lấy biển báo màu đỏ 
def returnRedness(img): 
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return v

# Áp dụng ngưỡng nhị phân cho hình ảnh.
def threshold(img,T=150):
	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 
# Làm thành ảnh đen trắng
def morphology(img,kernelSize=7):
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return opening

# Tìm đường viền trong ảnh
def findContour(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

# Tìm đường viền ngoài cùng lớn nhất trong ảnh
def findBiggestContour(contours):
    if not contours:
        return None
    c = [cv2.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

# Vẽ lên ảnh được tìm thấy
def boundaryBox(img,contours):
	x,y,w,h=cv2.boundingRect(contours)
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	sign=img[y:(y+h) , x:(x+w)]
	return img,sign
    
# lấy ra ảnh được tìm thấy
def boundaryBoxNew(img, contours):
    x, y, w, h = cv2.boundingRect(contours)
    img_vung_chon = img[y:y + h, x:x + w].copy()
    return img_vung_chon

def loadModelPredict(img):
    image_fromarray = Image.fromarray(img, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255
    pred = loaded_model.predict(input_data)
    result = pred.argmax()
    prediction = ""
    if result == 0:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 20km/h
    elif result == 1:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 30km/h
    elif result == 2:
        prediction = 'Giới hạn tốc độ (50km/h)'
    elif result == 3:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 60km/h
    elif result == 4:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 70km/h
    elif result == 5:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 80km/h
    elif result == 6:
        prediction = 'Hết giới hạn tốc độ' # 80km
    elif result == 7:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 100km/h
    elif result == 8:
        prediction = 'Cẩn thận. Có biển giới hạn tốc độ' # 120km/h
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
        prediction = 'Phía trước rẽ phải'
    elif result == 34:
        prediction = 'Phía trước rẽ trái'
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
    return result, prediction

@socketio.on('video_frame')
def  handle_video_frame(frame_data):
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Kiểm tra xem img chuẩn chưa đã, vì có lúc socketio rớt tín hiệu ko gửi đến
    if isinstance(frame, np.ndarray):
        img = filteringImages(frame)        
        #  Chuyển đổi hình ảnh sang không gian màu YUV và trả về kênh "v or v", nhấn mạnh vào màu đỏ và xanh của biển báo
        img = returnRedAndBlue(img)
        # Áp dụng ngưỡng nhị phân cho hình ảnh.
        img = threshold(img,T = 150)
        # Đổi ảnh đen trắng
        img=morphology(img,11)
        # check ảnh có hình 3 cạnh, 4 cạnh, tròn
        checkBienBao = is_circle_or_triangle(img)
        # Tìm đường viền trong ảnh nhị phân.
        contours = findContour(img)
        # Xác định đường viền lớn nhất trong danh sách các đường viền.
        big = findBiggestContour(contours)
        if big is not None and cv2.contourArea(big) > 100:
            img = boundaryBoxNew(frame, big) # cut những ảnh là biển báo khi quay video đi đường    
            if(checkBienBao == 1):
                # Lưu lại làm store sau check
                file_name = 'img/bien_bao_' + str(int(time.time() * 1000)) + '.jpg'
                cv2.imwrite(file_name, img)     
            # bắn thông báo tới client
            result, prediction = loadModelPredict(img)
            if(result < 9):
                # print(prediction)
                # truyền data tới tất cả các máy khách được kết nối nếu cần
                emit('video_frame', prediction, broadcast=True)

if __name__ == '__main__':
    # Use eventlet to run the application with SSL
    import eventlet
    import eventlet.wsgi
    from eventlet import wrap_ssl
    
    eventlet.wsgi.server(wrap_ssl(eventlet.listen(('0.0.0.0', 5000)), certfile='server.crt', keyfile='server.key', server_side=True),
                        app,
                        debug=True,
                        log_output=False)