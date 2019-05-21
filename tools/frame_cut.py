import cv2
import os
import glob

curr_path = os.path.realpath(__file__)

total_num = 0
for id in range(1,9):
    videos_path =r'D:\orange_pic\videos/{}/{}.mp4'.format(id,id)
    save_dir_path = os.path.dirname(videos_path)

    cap = cv2.VideoCapture(videos_path)
    dir_path = os.path.dirname(videos_path)
    assert cap.isOpened(),"camera not connected or video not exist"
    frame_count =0
    pic_num =0

    try:
        while(cap.isOpened()):

            _,frame = cap.read()
            # cv2.imshow('pic',frame)
            frame_count +=1
            if(frame_count ==30):
                frame_count =0
                print('id:{} orange pic: {}'.format(id,pic_num))
                cv2.imshow('pic',frame)
                cv2.imwrite(os.path.join(save_dir_path,'{}_{}.jpg').format(id,pic_num),frame)
                pic_num += 1
            k=cv2.waitKey(1)
            if k&0xff == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    except Exception:
        print("total pic num: {}".format(pic_num))
        total_num += pic_num
        continue
print('total num: {}'.format(total_num))