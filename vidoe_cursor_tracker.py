import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity  
from skimage import measure, morphology
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, footprint_rectangle
from pprint import pprint

import os
from dotenv import load_dotenv

load_dotenv()

def video_reader(video, sample_rate=0):
    """video iterator, yielding frame and frame end timestamp in ms"""
    if isinstance(video, str):
        source = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        source = video
    else:
        raise ValueError(f'"video" argument should be str-path (link) to video or cv2.VideoCapture, got {type(video)}')

    success = source.grab()
    frame_number = 0
    while success:
        if frame_number % (sample_rate + 1) == 0:
            _, frame = source.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame, source.get(cv2.CAP_PROP_POS_MSEC), frame_number

        frame_number += 1
        success = source.grab()

video = os.getenv("VIDEO_PATH")
if video is None:
    raise ValueError("Переменная VIDEO_SOURCE не найдена в .env!")
    
video = cv2.VideoCapture(video)

fps = video.get(cv2.CAP_PROP_FPS)
prev = None
for frame, timestamp, frame_number in video_reader(video,fps// 128):
    # print(frame, timestamp, frame_number)
    cv2.imwrite(f"./openCV_test/test_frames/{frame_number}.png", frame)
    
    if not isinstance(prev, type(None)):
        (score, diff) = structural_similarity(frame, prev, full=True)
        
        
        print("Image Similarity: {:.4f}%".format(score * 100))
        if score*100 > 99:

            my_diff = np.where(abs(prev-frame)>10, 0, 255).astype(np.uint8)
            cv2.imwrite(f"./openCV_test/test_frames/{frame_number}_0.png", my_diff)

            gray = 255 - cv2.absdiff(prev, frame)
            cv2.imwrite(f"./openCV_test/test_frames/{frame_number}_1.png", gray)
            
            
            gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

            diff = cv2.GaussianBlur(gray, (3, 3), 0)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite(f"./openCV_test/test_frames/{frame_number}_2.png", diff)

            labels = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)[1]

            #cv2.imshow("frame", labels)
            #cv2.waitKey()

            labels = measure.label(labels, connectivity=2, background=255)
            regions = measure.regionprops(labels)

            cursor_regions = []
            for r in regions:
                minr, minc, maxr, maxc = r.bbox
                h = maxr - minr
                w = maxc - minc

                if 16 <= h <= 60 and 16 <= w <= 60 and r.area >= 20:
                    if (h/w < 3 if h/w > 1 else w/h < 3):
                        cursor_regions.append(r)

            print("Найдено курсоров:", len(cursor_regions))

            plt.figure(figsize=(12, 4))
            plt.title("Labeled Regions")
            plt.imshow(labels, cmap="nipy_spectral")
            plt.axis("off")
            plt.show()

            for i, r in enumerate(cursor_regions, start=1):
                minr, minc, maxr, maxc = r.bbox

                pad = 2
                minr = max(0, minr - pad)
                minc = max(0, minc - pad)
                maxr = min(gray.shape[0], maxr + pad)
                maxc = min(gray.shape[1], maxc + pad)

                template = gray[minr:maxr, minc:maxc].copy()

                # маска региона по labels
                region_mask = (labels[minr:maxr, minc:maxc] == r.label).astype(np.uint8) * 255

                # Очистка внешнего белого фона 
                if len(template.shape) == 2:
                    template_gray = template
                    template_bgr = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                else:
                    template_bgr = template
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                h_t, w_t = template_gray.shape[:2]

                # Почти белые пиксели
                white_mask = (template_gray >= 245).astype(np.uint8) * 255

                # Flood fill внешнего белого фона от углов
                flood = white_mask.copy()

                for seed_x, seed_y in [(0, 0), (w_t - 1, 0), (0, h_t - 1), (w_t - 1, h_t - 1)]:
                    if flood[seed_y, seed_x] == 255:
                        ff_mask = np.zeros((h_t + 2, w_t + 2), dtype=np.uint8)
                        cv2.floodFill(flood, ff_mask, (seed_x, seed_y), 128)

                outer_bg = (flood == 128)

                #cv2.imshow("frame", flood)
                #cv2.waitKey()



                alpha = np.where((~outer_bg), 255, 0).astype(np.uint8)

                # Можно ещё почистить template по alpha:
                cleaned_bgr = template_bgr.copy()
                cleaned_bgr[alpha == 0] = 0

                rgba = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = alpha 

                # Сохраняем
                cv2.imwrite(f"cursor_template_{frame_number}_{i}.png", rgba)
                cv2.imwrite(f"cursor_mask_{frame_number}_{i}.png", alpha)

                print(f"cursor {i}: bbox=({minc}, {minr})-({maxc}, {maxr})")

            
            

            


    prev = frame
    if frame_number >80:
        break
