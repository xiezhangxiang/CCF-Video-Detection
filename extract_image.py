# -*- coding: utf-8 -*-
"""
# @Author  : xiezhangxiang
# @Email   : whitexiezx@gmail.com
"""
import os

import cv2 as cv
from tqdm import tqdm
import glob
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import math


def get_frames(video_holder, nps, save_path, video_src_folder):
    for video_name in video_holder:
        video_folder = os.path.join(save_path, video_name)
        if not os.path.exists(video_folder):
            idx = 1
            os.mkdir(video_folder)
            print(video_name)
            video_path = os.path.join(video_src_folder, video_name + '.mp4')
            cap = cv.VideoCapture(video_path)
            fps = cap.get(cv.CAP_PROP_FPS)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame_ms = int(idx / fps * 1000)
                if (idx-1) % nps == 0:
                    cv.imwrite('{0}/{1}_{2}_{3}.jpg'.format(video_folder, video_name, idx, frame_ms), frame)
                idx = idx + 1

            cap.release()


def get_imgholder(video_list, core_n):
    img_holder = []
    batch_size = math.ceil(len(video_list) / core_n)
    for i in range(core_n):
        if len(video_list) < batch_size:
            img_holder.append(video_list[:])
            del video_list[:]
        else:
            img_holder.append(video_list[:batch_size])
            del video_list[:batch_size]
    return img_holder


def process(holder, nps, save_folder, video_folder):
    process_list = []
    for i in range(multicore_n):
        args = (holder[i], nps, save_folder, video_folder)
        p = mp.Process(target=get_frames, args=args)
        process_list.append(p)
    for i in range(multicore_n):
        process_list[i].start()
    for i in range(multicore_n):
        process_list[i].join()


if __name__ == "__main__":

    query_video_folder = '/data/query'  # query视频存放路径
    refer_video_folder = '/data/refer'  # refer视频存放路径

    refer_list = [e.replace('.mp4', '') for e in os.listdir(refer_video_folder) if e.endswith('.mp4')]
    query_list = [e.replace('.mp4', '') for e in os.listdir(query_video_folder) if e.endswith('.mp4')]

    refer_save_folder = '/data/refer_frames'  # refer视频帧存放路径
    query_save_folder = '/data/query_frames'  # query视频帧存放路径

    if not os.path.exists(refer_save_folder):
        os.mkdir(refer_save_folder)

    if not os.path.exists(query_save_folder):
        os.mkdir(query_save_folder)

    refer_list_processed = os.listdir(refer_save_folder)
    query_list_processed = os.listdir(query_save_folder)

    query_videos = list(set(query_list) - set(query_list_processed))
    refer_videos = list(set(refer_list) - set(refer_list_processed))

    multicore_n = 4  # 进程个数
    # frame_interval = 5  # 抽帧间隔

    query_holder = get_imgholder(query_videos, multicore_n)
    refer_holder = get_imgholder(refer_videos, multicore_n)

    process(query_holder, 5, query_save_folder, query_video_folder)
    process(refer_holder, 10, refer_save_folder, refer_video_folder)

