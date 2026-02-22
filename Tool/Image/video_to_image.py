import cv2

def process_video_by_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    c = 1
    frameRate = 1  # 帧数截取间隔（每隔1帧截取一帧）
    
    while(True):
        ret, frame = cap.read()
        if ret:
            if(c % frameRate == 0):
                print("开始截取视频第：" + str(c) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                cv2.imwrite(save_path + str(c) + '.png', frame)  # 这里是将截取的图像保存在本地
            c += 1
            cv2.waitKey(0)
        else:
            print("所有帧都已经保存完成")
            break
    cap.release()

def process_video_by_time(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(5)  # 这个是获取视频帧率

    cap = cv2.VideoCapture(video_path)
    c = 1
    timeRate = 1  # 截取视频帧的时间间隔（这里是每隔10秒截取一帧）
    
    while(True):
        ret, frame = cap.read()
        FPS = cap.get(5)
        if ret:
            frameRate = int(FPS) * timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
            if(c % frameRate == 0):
                print("开始截取视频第：" + str(c) + " 帧")
                # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
                cv2.imwrite(save_path + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
            c += 1
            cv2.waitKey(0)
        else:
            print("所有帧都已经保存完成")
            break
    cap.release()

if __name__ == '__main__':
    video_path = '../Dataset/Video/UoB_IR_Tropical_2.1_i_Subclip_registered.avi'
    save_path = './Infrared/'
    # 根据指定帧数抓帧
    process_video_by_frame(video_path, save_path)
    # 根据指定时间抓帧
    # process_video_by_time(video_path, save_path)