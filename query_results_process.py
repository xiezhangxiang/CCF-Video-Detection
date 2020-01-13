# -*- coding: utf-8 -*-
"""
# @Author  : xiezhangxiang
# @Email   : whitexiezx@gmail.com
"""
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='gen video pair list')

parser.add_argument('--query_results_foler', type=str,
                    default='../CCF_Video_Detection/train_retrieve_results',
                    help='query results dir')
parser.add_argument('--threshold', type=float, default=0.03, help='video pair threshold')

args = parser.parse_args()

if __name__ == '__main__':
    notOnlyTop1 = True
    train_query = True
    threshold = 0.00
    rest_num = 0
    query_results_foler = '../CCF_Video_Detection/train_retrieve_results'
    # query_results_foler = '../CCF_Video_Detection/test_data2/retrieve_results2'
    cdva_results_dict = dict()

    for csv_file in glob.glob('%s/*.csv' % query_results_foler):
        rs = pd.read_csv(csv_file)
        query_list = rs[' queryID']
        refer_list = rs[' matchingReferenceID']
        score_list = rs[' score']
        for i in range(len(query_list)):
            query_id = query_list[i].split('/')[-1].replace('.mp4', '')
            refer_id = refer_list[i].split('/')[-1].replace('.mp4', '')
            if query_id not in cdva_results_dict.keys():
                cdva_results_dict[query_id] = [[refer_id, score_list[i]]]
            else:
                top1_score = cdva_results_dict[query_id][0][1]
                if score_list[i] > top1_score:
                    cdva_results_dict[query_id][0] = [refer_id, score_list[i]]
                elif top1_score - score_list[i] <= threshold and notOnlyTop1 is True:
                    cdva_results_dict[query_id].append([refer_id, score_list[i]])
                    rest_num += 1

    if train_query:
        ground_truth = pd.read_csv('../CCF_Video_Detection/train_data/train.csv')
        # print(ground_truth)
        query_id_gt_list = ground_truth['query_id']
        refer_id_gt_list = ground_truth['refer_id']

        right = 0.0
        total = 0.0

        query_path = 'train_data/query'
        refer_path = 'train_data/linux5.2/refer'

        # with open('/home/xiezhangxiang/Projects/CDVA/CCF_Video_Detection/train_retrieve_results/train_query_top1/top1.txt', 'a') as ftop1:
        #     with open('/home/xiezhangxiang/Projects/CDVA/CCF_Video_Detection/train_retrieve_results/train_query_0.05/top1_0.05.txt', 'a') as fthreshold:
                # print(cdva_results_dict.keys())
        for i in range(len(query_id_gt_list)):
            query_id = query_id_gt_list[i]
            refer_id_gt = refer_id_gt_list[i]
            if query_id in cdva_results_dict.keys():
                total += 1
                cdva_refer_id_list = [refer_id[0] for refer_id in cdva_results_dict[query_id]]
                print(query_id, refer_id_gt, cdva_refer_id_list)
                if str(refer_id_gt) in cdva_refer_id_list:
                    right += 1

                            # if len(cdva_results_dict[query_id]) == 1:
                            #     line = '{}/{}.mp4 {}/{}.mp4\n'.format(query_path, query_id, refer_path, cdva_results_dict[query_id][0][0])
                            #     ftop1.writelines(line)
                            # else:
                            #     for value in cdva_results_dict[query_id]:
                            #         line = '{}/{}.mp4 {}/{}.mp4\n'.format(query_path, query_id, refer_path,
                            #                                             value[0])
                            #         fthreshold.writelines(line)

        print('acc:', right / total, 'right: ', right, 'total:', total, 'rest_num:', rest_num)
    else:
        print('total match list len:', len(cdva_results_dict.values()))
        with open('query_match_list_{}threshold.txt'.format(args.threshold), 'a') as f:
            for query_id in cdva_results_dict.keys():
                for e in cdva_results_dict[query_id]:
                    line = str(query_id) + ' ' + str(e[0]) + '\n'
                    f.writelines(line)
