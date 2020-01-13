# -*- coding: utf-8 -*-
"""
# @Author  : xiezhangxiang
# @Email   : whitexiezx@gmail.com
"""
import cdvspy as cp
import cv2
import time

import os.path as osp
import os

import numpy as np
from utils import setup_logger

from tqdm import tqdm
import multiprocessing as mp

import numpy as np


class Matcher(mp.Process):
    def __init__(self, producer_Queue, ip_adr):
        super(Matcher, self).__init__()
        self.producer_Queue = producer_Queue
        self.q_size = producer_Queue.qsize()
        # cdvs extract server
        cp.set_remote_addr(ip_adr[0], ip_adr[1])
        # init database
        self.db = cp.IndexDB()

    def run(self):
        while True:
                if producer_Queue.empty():
                    break

                idx, query, refer = self.producer_Queue.get()

                results_path = osp.join(results_root, '{0}_{1}_scores.txt'.format(query, refer))

                query_imgs = os.listdir(osp.join(query_img_folder, query))
                refer_imgs = os.listdir(osp.join(refer_img_folder, refer))

                start = time.time()

                print('{}/{} Query: {} , Reference: {} start Generating match score'.format(idx, self.q_size, query, refer))

                # self.lock.acquire()
                # try:
                #     logger.info('{}/3000 Query: {} , Reference: {} start Generating match score'.format(idx, query, refer))
                # except Exception:
                #     print('write logger failed!')
                #     continue
                # finally:
                #     self.lock.release()

                self.match_record(query_imgs, refer_imgs, query, refer, results_path)

                time_spend = int((time.time() - start) / 60.0)

                print('Retrieval time : {} s'.format(time_spend))

                # self.lock.acquire()
                # try:
                #     logger.info('Retrieval time : {} s'.format(time_spend))
                # except Exception:
                #     print('write logger failed!')
                #     continue
                # finally:
                #     self.lock.release()

    def match_record(self, _query_imgs, _refer_imgs, _q_video, _r_video, _results_path):

        self.db.clear_records()

        # add reference images record to database
        # for refer_img in tqdm(_refer_imgs, total=len(_refer_imgs), leave=False):
        for refer_img in _refer_imgs:
            img_path = osp.join(refer_img_folder, _r_video, refer_img)

            try:
                img_mat = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (640, 480))

                start = time.time()
                img_fea = cp.getfeature_cuda(img_mat)

                idx_ms = refer_img.replace('.jpg', '').split('_')[-1]
                self.db.add_record(img_fea, str(idx_ms))

            except:
                continue

        # extract query image and do retrieval
        matched_dict = dict()
        # for query_img in tqdm(_query_imgs, total=len(_query_imgs), leave=False):
        for query_img in _query_imgs:
            img_path = osp.join(query_img_folder, _q_video, query_img)

            start = time.time()

            try:
                img_mat = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (640, 480))
                img_fea = cp.getfeature_cuda(img_mat)
            except:
                continue

            _res = self.db.retrieve2(img_fea, 100, 0.0, 500, -float('inf'))

            if len(_res) != 0:
                query_frame_idx = int(query_img.replace('.jpg', '').split('_')[-1])

                matched_dict[query_frame_idx] = _res

        # record the scores
        with open(_results_path, 'a') as f:
            for q in matched_dict.keys():
                for e in matched_dict[q]:
                    line = str(q) + ' ' + str(e[1]) + ' ' + str(np.around(float(e[0]), decimals=4)) + '\n'
                    f.writelines(line)


def main():
    n = 0
    for pair_line in qr_pairs:
        ll = pair_line.replace('\n', '').replace('\t', '').split(' ')
        query = ll[0]
        refer = ll[1]

        results_path = osp.join(results_root, '{0}_{1}_scores.txt'.format(query, refer))
        if os.path.exists(results_path):
            print('{}/{} Query: {} , Reference: {} already matched'.format(n, len(qr_pairs), query, refer))
            continue

        # add pair to queue
        n += 1
        producer_Queue.put((n, query, refer))

    matcher_list = [Matcher(producer_Queue, ip_adr) for _ in range(worker)]

    for m in matcher_list:
        m.start()

    for m in matcher_list:
        m.join()


if __name__ == '__main__':
    # logger = setup_logger('train_query_matching', 'logs/cdvs', 0)
    # logger.info('start matching')

    qr_pair_file = '/data/query_match_list_0.03threshold_7.txt'  # 视频对文件
    qr_pairs = open(qr_pair_file).readlines()

    query_img_folder = '/data/query_frames'  # query视频帧保存文件夹
    refer_img_folder = '/data/refer_frames'   # refer视频帧保存文件夹

    results_root = '/data/cdvs_scores'  # cdvs相似度保存文件夹

    ip_adr = ('127.0.0.1', 2207)  # 修改cdvs服务端口号

    producer_Queue = mp.Queue()

    worker = 4  # 开启的进程数

    main()
