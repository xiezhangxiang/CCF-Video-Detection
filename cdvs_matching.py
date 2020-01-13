# -*- coding: utf-8 -*-
"""
# @Author  : xiezhangxiang
# @Email   : whitexiezx@gmail.com
"""
import os

import numpy as np

from collections import defaultdict

import pandas as pd
import glob
import math

import copy


def windows_results(frams_idxs, stride, match_n):
    frams_idxs.sort()

    if stride > frams_idxs[-1] - frams_idxs[0] or len(frams_idxs) <= match_n:
        return frams_idxs

    t = frams_idxs[0] + stride
    windows_start = 0
    windows_end = len(frams_idxs) - 1

    start = 0
    end = len(frams_idxs) - 1

    while stride < frams_idxs[end] - frams_idxs[windows_start]:
        a = np.array(frams_idxs)
        windows_end = a[a <= t].shape[0]
        if windows_end - windows_start + 1 > match_n:
            # print('start is setting')
            start = windows_start
            break

        windows_start += 1
        t = min(frams_idxs[windows_start] + stride, frams_idxs[-1])

    t = frams_idxs[-1] - stride
    windows_end = len(frams_idxs) - 1

    while stride < frams_idxs[windows_end] - frams_idxs[start]:
        a = np.array(frams_idxs)
        windows_start = len(frams_idxs) - a[a >= t].shape[0]
        if windows_end - windows_start + 1 > match_n:
            end = windows_end
            # print('end is setting')
            break

        windows_end -= 1
        t = max(frams_idxs[start], frams_idxs[windows_end] - stride)

    return frams_idxs[start:end + 1]


def neighbour_clear(frames_idx_list):
    # 1. neighbourhood method
    del_array = np.empty(0, dtype=int)

    frames_idx_array = np.array(frames_idx_list, dtype=int)
    # logger.info('query:')
    for t in frames_idx_list:
        b = frames_idx_array[frames_idx_array != t]
        a = b[(b > t - 4125) & (b < t + 4125)]
        if len(a) < 5:
            del_array = np.append(del_array, t)
        else:
            pass
            # logger.info('t: {0}, e: {1}'.format(t, a))

    new_frame_idxs = np.setdiff1d(frames_idx_array, del_array)

    del del_array

    return list(new_frame_idxs)


def video_matching(results_path, threshold, onlyTop1=True):
    global right, avg_loss, total_loss, loss_n, id_wrong, pair_num

    results_list = open(results_path).readlines()

    query_video_id, refer_video_id, *_ = results_path.split('/')[-1].replace('.txt', '').split('_')
    print('Query: {} Refer: {}'.format(query_video_id, refer_video_id))

    matched_dict = defaultdict(list)

    for line in results_list:
        try:
            ll = line.replace('\n', '').replace('\t', '').split(' ')

            if float(ll[2]) > threshold:
                if onlyTop1:
                    if int(ll[0]) not in matched_dict.keys():
                        matched_dict[int(ll[0])].append((int(ll[1]), float(ll[2])))
                    elif float(ll[2]) > matched_dict[int(ll[0])][0][1]:
                        matched_dict[int(ll[0])][0] = (int(ll[1]), float(ll[2]))

                else:
                        matched_dict[int(ll[0])].append((int(ll[1]), float(ll[2])))
        except:
            continue

    # 1. neighbourhood function
    query_start_ends = neighbour_clear(list(matched_dict.keys()))
    refer_start_ends = neighbour_clear([idx[0] for idxs in matched_dict.values() for idx in idxs])

    if len(query_start_ends) != 0 and len(refer_start_ends) != 0:
        # 2. matching window function
        query_start_end_time = windows_results(query_start_ends, 30000, 24)  # match num in 30s > 24
        refer_start_end_time = windows_results(refer_start_ends, 30000, 24)  # match num in 30s > 24

        q_start = min(query_start_end_time)
        q_end = max(query_start_end_time)

        r_start = min(refer_start_end_time)
        r_end = max(refer_start_end_time)

        print('Query video time: {} -> {}'.format(q_start, q_end))
        print('Reference video time: {} -> {}'.format(r_start, r_end))

        if TEST:
            submit_df = pd.read_csv('../CCF_Video_Detection/test_data2/result.csv')
            submit_df.loc[submit_df['query_id'] == query_video_id,
                          ['query_time_range(ms)', 'refer_id', 'refer_time_range(ms)']] \
                = ['{0}|{1}'.format(min(query_start_end_time), max(query_start_end_time)), refer_video_id,
                   '{0}|{1}'.format(min(refer_start_end_time), max(refer_start_end_time))]
            submit_df.to_csv('../CCF_Video_Detection/test_data2/result.csv', index=None, sep=',')
        else:
            idx = list(ground_truth['query_id']).index(query_video_id)

            loss = 0.0
            if int(ground_truth['refer_id'][idx]) == int(refer_video_id):
                qt = [int(e) for e in ground_truth['query_time_range(ms)'][idx].split('|')]
                rt = [int(e) for e in ground_truth['refer_time_range(ms)'][idx].split('|')]
                loss = math.fabs(qt[0] - q_start) + math.fabs(qt[1] - q_end) + math.fabs(rt[0] - r_start) + math.fabs(rt[1] - r_end)

                loss_n += 1
                if loss < 3000:
                    right += 1

                total_loss += loss
            else:
                id_wrong += 1
                # line = '{} {} gt:{}\n'.format(query_video_id, refer_video_id, ground_truth['refer_id'][idx])
                # fw.writelines(line)
            print('loss_now={0} , loss_ave={1}, Acc={2}, right={3}, id_wrong={4}, temporal align={5}'.format(loss / 1000.0, total_loss / float(loss_n) / 1000,
                                                                                                   right / float(k),
                                                                                                   right,
                                                                                         id_wrong,
                                                                                                             right/float(loss_n)))


def get_pair_score(qid, rid, threshold):
    # 6507a6f8-b918-11e9-ad99-fa163ee49799_2447052500_scores.txt
    txt = '{}/{}_{}_scores.txt'.format(score_root, qid, rid)
    if not os.path.exists(txt):
        # print('{} not exitst!!!'.format(txt))
        return 0
        # raise Exception
    else:
        score = 0
        results_list = open(txt).readlines()
        for line in results_list:
            try:
                ll = line.replace('\n', '').replace('\t', '').split(' ')

                if float(ll[2]) > threshold:
                    score += float(ll[2])
            except:
                continue

        return score


if __name__ == '__main__':
    # temporal alignment
    TEST = True

    pair_txt = ''
    pairs_scores = defaultdict(list)

    if not TEST:
        ground_truth = pd.read_csv('../CCF_Video_Detection/train_data/train.csv')
        right = 0
        loss = 0.0
        total_loss = 0.0
        avg_loss = 0.0
        loss_n = 0
        id_wrong = 0
        score_root = '../cdvs_train_scores'

        # fw = open('false_train_pairs.txt', 'a')
        all_pairs = open('train_match_list_0.03threshold.txt').readlines()

        for pair_line in all_pairs:
            ll = pair_line.replace('\n', '').split(' ')
            qid = ll[0]
            rid = ll[1]
            pairs_scores[qid].append((get_pair_score(qid, rid, 0.22), rid))

        pair_num = len(pairs_scores.keys())
        k = 0
        for qid in pairs_scores.keys():
            k += 1
            print('\n{}/{}:'.format(k, pair_num))
            if sum([e[0] for e in pairs_scores[qid]]) == 0:
                rid = pairs_scores[qid][0][1]
                print(pairs_scores[qid])
                score_file = '{}/{}_{}_scores.txt'.format(score_root, qid, rid)
                video_matching(score_file, 0.1)
            else:
                pairs_scores[qid].sort()
                rid = pairs_scores[qid][-1][1]
                print(pairs_scores[qid])
                score_file = '{}/{}_{}_scores.txt'.format(score_root, qid, rid)
                video_matching(score_file, 0.1)
    else:
        score_root = '/data/cdvs_scores'
        all_pairs = open('query_match_list_0.03threshold.txt').readlines()

        for pair_line in all_pairs:
            ll = pair_line.replace('\n', '').split(' ')
            qid = ll[0]
            rid = ll[1]
            pairs_scores[qid].append((get_pair_score(qid, rid, 0.22), rid))

        pair_num = len(pairs_scores.keys())

        k = 0
        for qid in pairs_scores.keys():
            k += 1
            print('\n{}/{}:'.format(k, pair_num))
            pairs_scores[qid].sort()
            rid = pairs_scores[qid][-1][1]
            score_file = '{}/{}_{}_scores.txt'.format(score_root, qid, rid)
            video_matching(score_file, 0.06900000000000002)  # 0.1

        print('\nPair number:{}'.format(pair_num))


