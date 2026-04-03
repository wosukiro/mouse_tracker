import cv2 
import numpy as np
import time

def video_reader(video, sample_rate=0):
    if isinstance(video, str):
        source = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        source = video
    else:
        raise ValueError('Invalid video source')

    success = source.grab()
    frame_number = 0
    while success:
        if frame_number % (sample_rate + 1) == 0:
            _, frame = source.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame, source.get(cv2.CAP_PROP_POS_MSEC), frame_number

        frame_number += 1
        success = source.grab()

video_path = '/home/wosukiro/University/proctoring_ml/openCV_test/test_1.mp4'
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)


template_bgr = cv2.imread("/home/wosukiro/University/proctoring_ml/openCV_test/left_ptr_002.png")
template_gray_orig = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

template_click_bgr = cv2.imread("/home/wosukiro/University/proctoring_ml/openCV_test/hand2_002.png")
template_click_gray_orig = cv2.cvtColor(template_click_bgr, cv2.COLOR_BGR2GRAY)

template_I_bgr = cv2.imread("/home/wosukiro/University/proctoring_ml/openCV_test/xterm_002.png")
template_I_gray_orig = cv2.cvtColor(template_I_bgr, cv2.COLOR_BGR2GRAY)

templates = []
for i in range(15, 32, 2):
    t_gray = cv2.resize(template_gray_orig, (i, i))
    t_click_gray = cv2.resize(template_click_gray_orig, (i, i))
    t_I_gray = cv2.resize(template_I_gray_orig, (i, i))
    
    # Преобразование курсоров в границы
    t_edges = cv2.Canny(t_gray, 100, 200)
    templates.append((t_edges, i))

    t_click_edges = cv2.Canny(t_click_gray, 100, 200)
    templates.append((t_click_edges, i))

    t_I_edges = cv2.Canny(t_I_gray, 100, 200)
    templates.append((t_I_edges, i))

# 0.3-0.4 норма, если увеличить коэф, то есть шанс, что курсор просто не будет найден
threshold = 0.35 
cursor_on_last_frame = False 

for frame, timestamp, frame_number in video_reader(video, int(fps) // 128):
    if frame_number < 19:
        continue
    start_time = time.time()
    
    # Превращаем текущий кадр видео в границы
    frame_edges = cv2.Canny(frame, 100, 200)
    cv2.imwrite(f"./edges/{frame_number}_frame_edges.png",frame_edges)
    
    # Возвращаемся к поиску максимума
    best_match_val = -1 
    best_match_loc = None
    best_match_size = None

    for t_edges, size in templates:
        res = cv2.matchTemplate(frame_edges, t_edges, cv2.TM_CCOEFF_NORMED)
            
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Курсор один - ищем большее совпадение
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_match_size = size
            

    new_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if best_match_val >= threshold:

        cursor_on_last_frame = True

        top_left = best_match_loc
        h, w = best_match_size, best_match_size
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv2.rectangle(new_img, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imwrite(f'./openCV_test/squared_cursor/{frame_number}_res.png', new_img)

    else: 

        cursor_on_last_frame = False
    
    end_time = time.time()
    print(f"Кадр {frame_number}, время: {end_time - start_time:.4f} с, уверенность: {best_match_val:.4f}")

    if frame_number > 30:
        break