from concurrent.futures.thread import _threads_queues
import json
import random
from functools import partial
import pdb
from turtle import pd
import numpy as np
import redis
import sklearn
import torch
# print(torch.__version__)
from eveliver import (Logger, load_model, tensor_to_obj)
from trainer import Trainer, TrainerCallback
from transformers import AutoTokenizer, BertModel, AutoModel
from matrix_transformer import Encoder as MatTransformer
from graph_encoder import Encoder as GraphEncoder
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from buffer import Buffer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME, contrastive_pair, check_htb_debug, complete_h_t_debug
from utils import complete_h_t, check_htb, check_htb_debug
from utils import CLS_TOKEN_ID, SEP_TOKEN_ID, H_START_MARKER_ID, H_END_MARKER_ID, T_END_MARKER_ID, T_START_MARKER_ID
import math
from torch.nn import CrossEntropyLoss
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from itertools import groupby
from pyg_graph import create_edges, create_graph, GCN, Attention, create_graph_single
from utils import DotProductSimilarity
from sentence_reordering import SentReOrdering
from sbert_wk import sbert
from itertools import product, combinations
from layers import GraphConvolution, GraphAttentionLayer, GRUCell, SGRU
from sklearn import metrics
from sklearn import metrics


def eval_performance_sklearn(predict, truelabel):
    f1 = metrics.f1_score(truelabel, predict, average='micro')
    fpr, tpr, thresholds = metrics.roc_curve(truelabel, predict, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return {'f1': f1, "auc": auc}


def eval_performance(facts, pred_result):
    sorted_pred_result_top500 = None
    sorted_pred_result_top1k = None
    p100 = 0
    p200 = 0
    p300 = 0
    p500 = 0
    p1k = 0
    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec = []
    prec_500 = []
    prec_1k = []
    rec = []
    correct = 0
    total = len(facts)
    # pdb.set_trace()

    for i, item in enumerate(sorted_pred_result):
        if (item['entpair'][0], item['entpair'][1], item['relation']) in facts:
            correct += 1
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))
        if i == 100:
            p100 = correct / (i + 1)
        if i == 200:
            p200 = correct / (i + 1)
        if i == 300:
            p300 = correct / (i + 1)
        if i == 500:
            p500 = correct / (i + 1)
        if i == 1000:
            p1k = correct / (i + 1)
    auc = sklearn.metrics.auc(x=rec, y=prec)
    np_prec = np.array(prec)

    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()
    return {'prec': np_prec.tolist(), 'rec': np_rec.tolist(), 'mean_prec': mean_prec, 'f1': f1, 'auc': auc,
            'p100': p100, 'p200': p200, 'p300': p300,
            'p500': p500, 'p1k': p1k}


def expand(start, end, total_len, max_size):
    e_size = max_size - (end - start)
    _1 = start - (e_size // 2)
    _2 = end + (e_size - e_size // 2)
    if _2 - _1 <= total_len:
        if _1 < 0:
            _2 -= -1
            _1 = 0
        elif _2 > total_len:
            _1 -= (_2 - total_len)
            _2 = total_len
    else:
        _1 = 0
        _2 = total_len
    return _1, _2


def place_train_data(dataset):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, 'n/a', l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            for label in labels:
                if label != 'n/a':
                    ds = l2docs[label]
                    if 'n/a' in l2docs:
                        ds.extend(l2docs['n/a'])
                    bags.append([key, label, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + x[1])
    return bags


def place_dev_data(dataset, single_path):
    ep2d = dict()
    for key, doc1, doc2, label in dataset:
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags


def place_test_data(dataset, single_path):
    ep2d = dict()
    for data in dataset:
        key = data['h_id'] + '#' + data['t_id']
        doc1 = data['doc'][0]
        doc2 = data['doc'][1]
        label = 'n/a'
        if key not in ep2d:
            ep2d[key] = dict()
        if label not in ep2d[key]:
            ep2d[key][label] = list()
        ep2d[key][label].append([doc1, doc2, label])
    bags = list()
    for key, l2docs in ep2d.items():
        if len(l2docs) == 1 and 'n/a' in l2docs:
            bags.append([key, ['n/a'], l2docs['n/a'], 'o'])
        else:
            labels = list(l2docs.keys())
            ds = list()
            for label in labels:
                if single_path and label != 'n/a':
                    ds.append(random.choice(l2docs[label]))
                else:
                    ds.extend(l2docs[label])
            if 'n/a' in labels:
                labels.remove('n/a')
            bags.append([key, labels, ds, 'o'])
    bags.sort(key=lambda x: x[0] + '#' + '#'.join(x[1]))
    return bags


def gen_c(tokenizer, passage, span, max_len, bound_tokens, d_start, d_end, tmpd_start, tmpd_end, no_additional_marker,
          mask_entity):
    ret = list()
    ret.append(bound_tokens[0])
    for i in range(span[0], span[1]):
        if mask_entity:
            ret.append('[MASK]')
        else:
            ret.append(passage[i])
    ret.append(bound_tokens[1])
    prev = list()
    prev_ptr = span[0] - 1

    while len(prev) < max_len:
        if prev_ptr < 0:
            break
        if not no_additional_marker and prev_ptr in d_end:
            prev.append(f'[unused{(d_end[prev_ptr] + 2) * 2 + 2}]')
        if not no_additional_marker and prev_ptr in tmpd_end:
            prev.append(f'*')

        prev.append(passage[prev_ptr])
        if not no_additional_marker and prev_ptr in d_start:
            prev.append(f'[unused{(d_start[prev_ptr] + 2) * 2 + 1}]')
        if not no_additional_marker and prev_ptr in tmpd_start:
            prev.append(f'*')

        prev_ptr -= 1

    nex = list()
    nex_ptr = span[1]
    while len(nex) < max_len:
        if nex_ptr >= len(passage):
            break
        if not no_additional_marker and nex_ptr in d_start:
            nex.append(f'[unused{(d_start[nex_ptr] + 2) * 2 + 1}]')
        if not no_additional_marker and nex_ptr in tmpd_start:
            nex.append(f'*')
        nex.append(passage[nex_ptr])
        if not no_additional_marker and nex_ptr in d_end:
            nex.append(f'[unused{(d_end[nex_ptr] + 2) * 2 + 2}]')
        if not no_additional_marker and nex_ptr in tmpd_end:
            nex.append(f'*')
        nex_ptr += 1
    prev.reverse()
    ret = prev + ret + nex

    return ret


def process(tokenizer, h, t, doc0, doc1):
    ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
    b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
    max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
    cnt, batches = 0, []
    d = []

    def fix_entity(doc, ht_markers, b_markers):
        markers = ht_markers + b_markers + ['*']
        markers_pos = []
        if list(set(doc).intersection(set(markers))):
            for marker in markers:
                try:
                    pos = doc.index(marker)
                    markers_pos.append((pos, marker))
                except ValueError as e:
                    continue

        idx = 0
        while idx <= len(markers_pos) - 1:
            try:
                assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (
                        int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) - int(
                    markers_pos[idx + 1][1].replace("[unused", "").replace("]", "")) == -1)
                entity_name = doc[markers_pos[idx][0] + 1: markers_pos[idx + 1][0]]
                while "." in entity_name:
                    assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    entity_name = doc[markers_pos[idx][0] + 1: markers_pos[idx + 1][0]]
                idx += 2
            except:
                # pdb.set_trace()
                idx += 1
                continue
        return doc

    d0 = fix_entity(doc0, ht_markers, b_markers)

    d1 = fix_entity(doc1, ht_markers, b_markers)

    for di in [d0, d1]:
        d.extend(di)
    d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
    d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
    dbuf = Buffer()
    dbuf.blocks = d0_buf.blocks + d1_buf.blocks
    for blk in dbuf:

        if list(set(tokenizer.convert_tokens_to_ids(ht_markers)).intersection(set(blk.ids))):
            blk.relevance = 2
        elif list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))) or list(
                set(tokenizer.convert_tokens_to_ids(['*'])).intersection(set(blk.ids))):
            blk.relevance = 1
        else:
            continue
    ret = []

    n0 = 1
    pbuf_ht, nbuf_ht = dbuf.filtered(lambda blk, idx: blk.relevance >= 2, need_residue=True)
    pbuf_b, nbuf_b = nbuf_ht.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)

    for i in range(n0):
        _selected_htblks = random.sample(pbuf_ht.blocks, min(max_blk_num, len(pbuf_ht)))
        _selected_pblks = random.sample(pbuf_b.blocks, min(max_blk_num - len(_selected_htblks), len(pbuf_b)))
        _selected_nblks = random.sample(nbuf_b.blocks,
                                        min(max_blk_num - len(_selected_pblks) - len(_selected_htblks), len(nbuf_b)))
        buf = Buffer()
        buf.blocks = _selected_htblks + _selected_pblks + _selected_nblks
        ret.append(buf.sort_())
    ret[0][0].ids.insert(0, tokenizer.convert_tokens_to_ids(tokenizer.cls_token))

    return ret[0]


def if_h_t_complete(buffer):
    h_flag = False
    t_flag = False
    h_markers = [1, 2]
    t_markers = [3, 4]
    for ret in buffer:
        if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
            continue
        else:
            if ret.ids.index(1) < ret.ids.index(2):
                h_flag = True
            else:
                continue
    for ret in buffer:
        if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
            continue
        else:
            if ret.ids.index(3) < ret.ids.index(4):
                t_flag = True
            else:
                continue
    if h_flag and t_flag:
        return True
    else:
        return False


def bridge_entity_based_filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk, doc_entities, dps_count):
    alpha = 1
    beta = 0.1
    gamma = 0.01
    K = 16

    def complete_h_t(all_buf, filtered_buf):
        h_markers = [1, 2]
        t_markers = [3, 4]
        for blk_id, blk in enumerate(filtered_buf.blocks):
            if blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
                if list(set(blk.ids).intersection(set(h_markers))) == [H_START_MARKER_ID]:
                    # pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p + 1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                elif list(set(blk.ids).intersection(set(h_markers))) == [H_END_MARKER_ID]:
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos - 2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
            elif blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers:
                # pdb.set_trace()
                markers_starts = []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == H_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == H_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p + 1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos - 2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

                else:
                    if blk.ids.index(H_END_MARKER_ID) > blk.ids.index(H_START_MARKER_ID):
                        pass
                    elif blk.ids.index(H_END_MARKER_ID) < blk.ids.index(H_START_MARKER_ID):
                        first_end_marker = blk.ids.index(H_END_MARKER_ID)
                        second_start_marker = blk.ids.index(H_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(H_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p + 1]
                        new = blk.ids + complementary
                        if len(new) <= 63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        # print(filtered_buf[blk_id].ids)

            elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
                if list(set(blk.ids).intersection(set(t_markers))) == [T_START_MARKER_ID]:
                    # pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p + 1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids).intersection(set(t_markers))) == [T_END_MARKER_ID]:
                    # pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos - 2].ids
                    marker_p = complementary.index(T_START_MARKER_ID)
                    marker_p_start = complementary.index(T_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]

                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

            elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers:
                # pdb.set_trace()
                markers_starts = []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == T_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == T_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p + 1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos - 2].ids
                    marker_p_start = complementary.index(3)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                else:
                    if blk.ids.index(T_END_MARKER_ID) > blk.ids.index(T_START_MARKER_ID):
                        pass
                    elif blk.ids.index(T_END_MARKER_ID) < blk.ids.index(T_START_MARKER_ID):
                        first_end_marker = blk.ids.index(T_END_MARKER_ID)
                        second_start_marker = blk.ids.index(T_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(T_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p + 1]
                        new = blk.ids + complementary
                        if len(new) <= 63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID:
                            filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
                        else:
                            continue
                        # print(filtered_buf[blk_id].ids)
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID and filtered_buf[blk_id].ids[0] not in [1, 3]:
                if len(filtered_buf[blk_id].ids) <= 63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    # pdb.set_trace()
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
            elif filtered_buf[blk_id].ids[0] in [1, 3]:
                if len(filtered_buf[blk_id].ids) <= 63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                    # pdb.set_trace()
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID or filtered_buf[blk_id].ids[-1] != SEP_TOKEN_ID:
                pdb.set_trace()
            else:
                pass
        return filtered_buf

    def detect_h_t(tokenizer, buffer):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        h_blocks = []
        t_blocks = []
        for blk in buffer:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                h_blocks.append(blk)
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                t_blocks.append(blk)
            else:
                continue
        return h_blocks, t_blocks

    def if_h_t_complete(buffer):
        h_flag = False
        t_flag = False
        h_markers = [1, 2]
        t_markers = [3, 4]
        for ret in buffer:
            if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
                continue
            else:
                if ret.ids.index(1) < ret.ids.index(2):
                    h_flag = True
                else:
                    if len(list(set(ret.ids).intersection(set([2])))) > len(list(set(ret.ids).intersection(set([1])))):
                        h_flag = True
                    else:
                        continue
        for ret in buffer:
            if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
                continue
            else:
                if ret.ids.index(3) < ret.ids.index(T_END_MARKER_ID):
                    t_flag = True
                else:
                    if len(list(set(ret.ids).intersection(set([T_END_MARKER_ID])))) > len(
                            list(set(ret.ids).intersection(set([3])))):
                        t_flag = True
                    else:
                        continue
        if h_flag and t_flag:
            return True
        else:
            return False

    def co_occur_graph(tokenizer, h, t, d0, d1, doc_entities, alpha, beta, gamma, dps_count):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []
        d = []

        for di in [d0, d1]:
            d.extend(di)
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
        dbuf = Buffer()
        dbuf.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf.blocks:
            if blk.ids[0] != CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids

        co_occur_pair = []
        for blk in dbuf:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))) and list(
                    set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idx = list(set([math.ceil(int(b_m) / 2) for b_m in
                                  list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))[0]
                co_occur_pair.append((1, b_idx, blk.pos))
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))) and list(
                    set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idx = list(set([math.ceil(int(b_m) / 2) for b_m in
                                  list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))[0]
                co_occur_pair.append((2, b_idx, blk.pos))
            elif list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids))):
                b_idxs = list(set([math.ceil(int(b_m) / 2) for b_m in
                                   list(set(tokenizer.convert_tokens_to_ids(b_markers)).intersection(set(blk.ids)))]))
                if len(b_idxs) >= 2:
                    pairs = combinations(b_idxs, 2)
                else:
                    pairs = []
                for pair in pairs:
                    co_occur_pair.append((pair[0], pair[1], blk.pos))
            else:
                continue

        h_co = list((filter(lambda pair: pair[0] == 1, co_occur_pair)))
        t_co = list((filter(lambda pair: pair[0] == 2, co_occur_pair)))
        b_co = list((filter(lambda pair: pair[0] > 2, co_occur_pair)))

        score_b = dict()
        s1 = dict()
        s2 = dict()
        s3 = dict()

        for entity_id in range(1, math.ceil((len(b_markers)) / 2) + 2):
            s1[entity_id] = 0
            s2[entity_id] = 0
            s3[entity_id] = 0
            score_b[entity_id] = 0

        for pair in co_occur_pair:
            if pair[0] <= 2:
                s1[pair[1]] = 1

        for pair in b_co:
            if s1[pair[0]] == 1:
                s2[pair[1]] += 1

            if s1[pair[1]] == 1:
                s2[pair[0]] += 1

        bridge_ids = {doc_entities[dps_count][key]: key for key in doc_entities[dps_count].keys()}
        for idx in range(len(doc_entities)):
            if idx == dps_count:
                continue
            else:
                ent_ids = doc_entities[idx].keys()
                for k, v in bridge_ids.items():
                    if v in ent_ids:
                        s3[k + 3] += 1
                    else:
                        continue

        for entity_id in range(1, math.ceil((len(b_markers)) / 2) + 2):
            score_b[entity_id] += alpha * s1[entity_id] + beta * s2[entity_id] + gamma * s3[entity_id]
        # pdb.set_trace()
        return score_b

    def get_block_By_sentence_score(tokenizer, h, t, d0, d1, score_b, K):
        # pdb.set_trace()
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []
        d = []

        score_b_positive = [(k, v) for k, v in score_b.items() if v > 0]
        score_b_positive_ids = []
        for b in score_b_positive:
            b_id = b[0]
            b_score = b[1]
            score_b_positive_ids.append(2 * b_id - 1)
            score_b_positive_ids.append(2 * b_id)

        # pdb.set_trace()
        for di in [d0, d1]:
            d.extend(di)
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, cnt=cnt, hard=False, docid=1)
        dbuf_all = Buffer()
        dbuf_all.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf_all.blocks:
            if blk.ids[0] != CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                blk.h_flag = 1
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                blk.t_flag = 1

        for blk in dbuf_all:
            if len(list(set(score_b_positive_ids).intersection(set(blk.ids)))) > 0:
                blk_bridge_marker_ids = list(set(score_b_positive_ids).intersection(set(blk.ids)))
                blk_bridge_ids = list(set([math.ceil(int(b_m_id) / 2) for b_m_id in blk_bridge_marker_ids]))
                for b_id in blk_bridge_ids:
                    blk.relevance += score_b[b_id]
                # print(blk.pos, blk.relevance)
            else:
                blk.relevance = 0
                continue

        # pdb.set_trace()
        for blk in dbuf_all:
            if blk.h_flag == 1 or blk.t_flag == 1:
                blk.relevance += 1
            else:
                continue

        block_scores = dict()
        for blk in dbuf_all:
            block_scores[blk.pos] = blk.relevance
            # print(blk.pos, blk.relevance)

        block_scores = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)

        try:
            score_threshold = block_scores[K][1]
        except IndexError as e:
            h_blocks = []
            t_blocks = []
            if not if_h_t_complete(dbuf_all):
                # pdb.set_trace()
                dbuf_all = complete_h_t(dbuf_all, dbuf_all)
                if not if_h_t_complete(dbuf_all):
                    pdb.set_trace()
                    dbuf_all = complete_h_t_debug(dbuf_all, dbuf_all)
            else:
                pass
            for blk in dbuf_all:
                if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                    h_blocks.append(blk)
                elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                    t_blocks.append(blk)
                else:
                    continue
            return h_blocks, t_blocks, dbuf_all, dbuf_all

        score_highest = block_scores[0][1]
        if score_threshold > 0 or score_highest > 0:
            p_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance > score_threshold, need_residue=True)
            e_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance == score_threshold, need_residue=True)
        else:
            p_buf, e_buf = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)

        if len(p_buf) + len(e_buf) == K:
            dbuf_filtered = p_buf + e_buf
        elif len(p_buf) + len(e_buf) < K:
            _, rest_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance < score_threshold, need_residue=True)
            dbuf_filtered = p_buf + e_buf + random.sample(rest_buf, K - len(p_buf) - len(e_buf))
            assert len(dbuf_filtered) <= K
        else:
            try:
                highest_blk_id = sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].pos
            except:
                if score_threshold > 0 or score_highest > 0:
                    highest_blk_id = sorted(e_buf, key=lambda x: x.relevance, reverse=True)[0].pos
                else:
                    detect_h_t(tokenizer, dbuf_filtered)
            e_buf_selected_blocks = []
            try:
                if sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].relevance > 0:
                    e_buf_distance = dict()
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = abs(e.pos - highest_blk_id)
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K - len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            except:
                if e_buf[0].relevance > 0:
                    e_buf_distance = dict()
                    ht_buf, _ = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in ht_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K - len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
                else:
                    e_buf_distance = dict()
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in p_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K - len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            dbuf_blocks = p_buf.blocks + e_buf_selected_blocks
            dbuf_filtered = Buffer()
            for block in dbuf_blocks:
                dbuf_filtered.insert(block)

        h_blocks = []
        t_blocks = []
        for blk in dbuf_filtered:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)).intersection(set(blk.ids))):
                h_blocks.append(blk)
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)).intersection(set(blk.ids))):
                t_blocks.append(blk)
            else:
                continue
        if len(h_blocks) == 0 or len(t_blocks) == 0:
            new_dbuf = Buffer()
            ori_dbuf_all_blocks = sorted(dbuf_all.blocks, key=lambda x: x.relevance * 0.01 + (x.h_flag + x.t_flag),
                                         reverse=True)
            ori_dbuf_filtered_blocks = sorted(dbuf_filtered.blocks,
                                              key=lambda x: x.relevance * 0.01 + (x.h_flag + x.t_flag), reverse=True)
            if len(h_blocks) == 0:
                candi_h_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.h_flag:
                        candi_h_blocks.append(blk)
                    else:
                        continue
                h_blocks.append(random.choice(candi_h_blocks))
                new_dbuf.insert(h_blocks[0])
            if len(t_blocks) == 0:
                candi_t_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.t_flag:
                        candi_t_blocks.append(blk)
                        # break
                    else:
                        continue
                t_blocks.append(random.choice(candi_t_blocks))
                new_dbuf.insert(t_blocks[0])
            for ori_blk in ori_dbuf_filtered_blocks:
                if len(new_dbuf) <= K - 1:
                    new_dbuf.insert(ori_blk)
                else:
                    break
            dbuf_filtered = new_dbuf

        h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
        all_block_pos = [blk.pos for blk in dbuf_filtered]
        if len(set(all_block_pos).intersection(set(h_t_block_pos))) != len(set(h_t_block_pos)):
            if len(set(all_block_pos).intersection(set(h_t_block_pos))) < len(set(h_t_block_pos)):
                h_blocks = [blk for blk in dbuf_filtered if blk.h_flag == 1]
                t_blocks = [blk for blk in dbuf_filtered if blk.t_flag == 1]
                h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
                assert len(set(all_block_pos).intersection(set(h_t_block_pos))) == len(set(h_t_block_pos))
            else:
                pdb.set_trace()
        if not if_h_t_complete(dbuf_filtered):
            dbuf_filtered = complete_h_t(dbuf_all, dbuf_filtered)
            if not if_h_t_complete(dbuf_filtered):
                pdb.set_trace()
                dbuf_filtered = complete_h_t_debug(dbuf_all, dbuf_filtered)

        else:
            pass
        return h_blocks, t_blocks, dbuf_filtered, dbuf_all

    score_b = co_occur_graph(tokenizer, h, t, doc0, doc1, doc_entities, alpha, beta, gamma, dps_count)
    h_blocks, t_blocks, dbuf, dbuf_all = get_block_By_sentence_score(tokenizer, h, t, doc0, doc1, score_b, K)
    if len(h_blocks) == 0 or len(t_blocks) == 0:
        pdb.set_trace()
    h_t_flag = False
    dbuf_concat = []
    for blk in dbuf:
        dbuf_concat.extend(blk.ids)
    h_t_flag = check_htb(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag)
    if not h_t_flag:
        pdb.set_trace()
        h_t_flag = check_htb_debug(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag)
    else:
        pass
    return h_blocks, t_blocks, dbuf, dbuf_all


def sent_Filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk, doc_entities, dps_count):
    def fix_entity(doc, ht_markers, b_markers):
        markers = ht_markers + b_markers
        markers_pos = []
        if list(set(doc).intersection(set(markers))):
            for marker in markers:
                try:
                    pos = doc.index(marker)
                    markers_pos.append((pos, marker))
                except ValueError as e:
                    continue

        idx = 0
        while idx <= len(markers_pos) - 1:
            try:
                assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (
                        int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) - int(
                    markers_pos[idx + 1][1].replace("[unused", "").replace("]", "")) == -1)
                entity_name = doc[markers_pos[idx][0] + 1: markers_pos[idx + 1][0]]
                while "." in entity_name:
                    assert doc[markers_pos[idx][0] + entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    entity_name = doc[markers_pos[idx][0] + 1: markers_pos[idx + 1][0]]
                idx += 2
            except:
                idx += 1
                continue
        return doc

    ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
    b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]

    doc0 = fix_entity(doc0, ht_markers, b_markers)
    doc1 = fix_entity(doc1, ht_markers, b_markers)

    h_blocks, t_blocks, dbuf, dbuf_all = bridge_entity_based_filter(tokenizer, h, t, doc0, doc1, encoder, sbert_wk,
                                                                    doc_entities, dps_count)

    sentence_blocks = dbuf.blocks
    block_pos = [blk.pos for blk in dbuf]
    order_start_blocks = [blk.pos for blk in h_blocks]
    order_end_blocks = [blk.pos for blk in t_blocks]
    if len(order_start_blocks) == 0 or len(order_end_blocks) == 0:
        pdb.set_trace()

    doc_0_blks = [blk for blk in sentence_blocks if blk.docid == 0]
    doc_1_blks = [blk for blk in sentence_blocks if blk.docid == 1]

    doc_0_sentences = [tokenizer.convert_ids_to_tokens(blk.ids) for blk in doc_0_blks]
    doc_1_sentences = [tokenizer.convert_ids_to_tokens(blk.ids) for blk in doc_1_blks]

    try:
        order_starts = [block_pos.index(pos) for pos in order_start_blocks]
        order_ends = [block_pos.index(pos) for pos in order_end_blocks]
    except:
        pdb.set_trace()

    for s in doc_0_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')
    for s in doc_1_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')
    # pdb.set_trace()
    sro = SentReOrdering(doc_0_sentences, doc_1_sentences, encoder=encoder, device='cuda', tokenizer=tokenizer, h=h,
                         t=t, sbert_wk=sbert_wk)
    orders = sro.semantic_based_sort(order_starts, order_ends)
    selected_buffers = []
    for order in orders:
        selected_buffer = Buffer()
        if len(order) <= 8:
            for od in order:
                try:
                    selected_buffer.insert(sentence_blocks[od])
                except Exception as e:
                    pdb.set_trace()
        else:
            # print(order)
            lll = 1
            o_scores = dict()
            for o in order[1:-1]:
                o_scores[o] = sentence_blocks[o].relevance
            o_scores = sorted(o_scores.items(), key=lambda s: s[1], reverse=True)
            while len(order) > 8:
                lowest_score = o_scores[-1][1]
                removable = list((filter(lambda o_score: o_score[1] == lowest_score, o_scores)))
                if len(removable) >= 1:
                    random.shuffle(removable)
                    remove_o = removable[0][0]
                order.remove(remove_o)
                o_scores.remove((remove_o, lowest_score))
            assert len(order) <= 8
            for od in order:
                try:
                    selected_buffer.insert(sentence_blocks[od])
                except Exception as e:
                    pdb.set_trace()
        selected_buffers.append(selected_buffer)
    # pdb.set_trace()
    return selected_buffers, dbuf_all


def process_example_ReoS(h, t, doc1, doc2, tokenizer, max_len, redisd, no_additional_marker, mask_entity, encoder,
                         sbert_wk, doc_entities, dps_count):
    max_len = 99999
    bert_max_len = 512
    doc1 = json.loads(redisd.get('codred-doc-' + doc1))
    doc2 = json.loads(redisd.get('codred-doc-' + doc2))
    v_h = None
    for entity in doc1['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
            v_h = entity
    assert v_h is not None
    v_t = None
    for entity in doc2['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
            v_t = entity
    assert v_t is not None
    d1_v = dict()
    for entity in doc1['entities']:
        if 'Q' in entity:
            d1_v[entity['Q']] = entity
    d2_v = dict()
    for entity in doc2['entities']:
        if 'Q' in entity:
            d2_v[entity['Q']] = entity

    ov = set(d1_v.keys()) & set(d2_v.keys())

    tmpov = set(d1_v.keys()) | set(d2_v.keys())

    ###The union minus the intersection gives us entities that are unique to each document.
    tmpov = tmpov - ov

    if len(ov) > 40:
        ov = set(random.choices(list(ov), k=40))
    ov = list(ov)
    ma = dict()
    tmpma = dict()

    for e in ov:
        ma[e] = len(ma)

    for e in tmpov:
        tmpma[e] = len(ma) + len(tmpma)

    d1_start = dict()
    d1_end = dict()
    tmp_d1_start = dict()
    tmp_d1_end = dict()

    for entity in doc1['entities']:
        if 'Q' in entity and entity['Q'] in ma:

            for span in entity['spans']:
                d1_start[span[0]] = ma[entity['Q']]
                d1_end[span[1] - 1] = ma[entity['Q']]

        if 'Q' in entity and entity['Q'] in tmpma:
            for span in entity['spans']:
                tmp_d1_start[span[0]] = tmpma[entity['Q']]
                tmp_d1_end[span[1] - 1] = tmpma[entity['Q']]

    d2_start = dict()
    d2_end = dict()
    tmp_d2_start = dict()
    tmp_d2_end = dict()

    for entity in doc2['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d2_start[span[0]] = ma[entity['Q']]
                d2_end[span[1] - 1] = ma[entity['Q']]

        if 'Q' in entity and entity['Q'] in tmpma:
            for span in entity['spans']:
                tmp_d2_start[span[0]] = tmpma[entity['Q']]
                tmp_d2_end[span[1] - 1] = tmpma[entity['Q']]

    k1 = gen_c(tokenizer, doc1['tokens'], v_h['spans'][0], max_len, ['[unused1]', '[unused2]'], d1_start, d1_end,
               tmp_d1_start, tmp_d1_end,
               no_additional_marker, mask_entity)
    k2 = gen_c(tokenizer, doc2['tokens'], v_t['spans'][0], max_len, ['[unused3]', '[unused4]'], d2_start, d2_end,
               tmp_d2_start, tmp_d2_end,
               no_additional_marker, mask_entity)

    selected_order_rets, dbuf_all = sent_Filter(tokenizer, v_h['name'], v_t['name'], k1, k2, encoder, sbert_wk,
                                                doc_entities, dps_count)

    if len(selected_order_rets) == 0:
        print("SELECTION FAIL")
        pdb.set_trace()
        return []
    else:
        pass
    # pdb.set_trace()
    h_flag = False
    t_flag = False
    h_markers = [1, 2]
    t_markers = [3, 4]
    for selected_order_ret in selected_order_rets:
        for ret in selected_order_ret:
            if list(set(ret.ids).intersection(set(h_markers))) != h_markers:
                continue
            else:
                h_flag = True
        for ret in selected_order_ret:
            if list(set(ret.ids).intersection(set(t_markers))) != t_markers:
                continue
            else:
                t_flag = True
        if h_flag and t_flag:
            pass
        else:
            pdb.set_trace()
            completed_selected_ret = complete_h_t(dbuf_all, selected_order_ret)
            if not if_h_t_complete(completed_selected_ret):
                completed_selected_ret = complete_h_t_debug(dbuf_all, selected_order_ret)
            selected_order_ret, dbuf_all = sent_Filter(tokenizer, v_h['name'], v_t['name'], k1, k2, encoder, sbert_wk,
                                                       doc_entities, dps_count)
            selected_order_rets[0] = completed_selected_ret

    return selected_order_rets[0]


def collate_fn(batch, args, relation2id, tokenizer, redisd, encoder, sbert_wk):
    sents_docs = []
    if batch[0][-1] == 'o':
        batch = batch[0]
        h, t = batch[0].split('#')
        r = relation2id[batch[1]]
        dps = batch[2]
        if len(dps) > 8:
            dps = random.choices(dps, k=8)
        dplabel = list()
        selected_rets = list()
        collec_doc1_titles = [doc1 for doc1, _, _ in dps]
        collec_doc2_titles = [doc2 for _, doc2, _ in dps]
        doc_entities = get_doc_entities(h, t, tokenizer, redisd, args.no_additional_marker, args.mask_entity,
                                        collec_doc1_titles, collec_doc2_titles)
        dps_count = 0
        for doc1, doc2, l in dps:
            selected_ret = process_example_ReoS(h, t, doc1, doc2, tokenizer, args.seq_len, redisd,
                                                args.no_additional_marker, args.mask_entity, encoder, sbert_wk,
                                                doc_entities, dps_count)
            for s_blk in selected_ret:
                while (tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                    s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(
                        ".")
            dplabel.append(relation2id[l])
            selected_rets.append(selected_ret)
        dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
        rs_t = torch.tensor([r], dtype=torch.int64)
        selected_inputs = torch.zeros(4, len(dps), CAPACITY, dtype=torch.int64)

        for dp, buf in enumerate(selected_rets):
            returns = buf.export_01_turn(out=(selected_inputs[0, dp], selected_inputs[1, dp], selected_inputs[2, dp]))
            sents_docs.append(returns[-1])
        selected_ids = selected_inputs[0]
        selected_att_mask = selected_inputs[1]
        selected_token_type = selected_inputs[2]
        selected_labels = selected_inputs[3]
    else:
        examples = batch[0]
        h_len = tokenizer.max_len_sentences_pair // 2 - 2
        t_len = tokenizer.max_len_sentences_pair - tokenizer.max_len_sentences_pair // 2 - 2
        _input_ids = list()
        _token_type_ids = list()
        _attention_mask = list()
        _rs = list()
        selected_rets = list()

        for idx, example in enumerate(examples):
            doc = json.loads(redisd.get(f'dsre-doc-{example[0]}'))
            _, h_start, h_end, t_start, t_end, r = example

            if r in relation2id:
                r = relation2id[r]
            else:
                r = 'n/a'
            h_1, h_2 = expand(h_start, h_end, len(doc), h_len)
            t_1, t_2 = expand(t_start, t_end, len(doc), t_len)

            docid = example[0]

            hentits = dict()
            tentits = dict()
            for j_idx, j_example in enumerate(examples):
                if idx != j_idx:
                    if j_example[0] == docid:
                        _, j_h_start, j_h_end, j_t_start, j_t_end, j_r = j_example
                        if j_h_start >= h_1 and j_h_end <= h_2:
                            hentits[" ".join(doc[j_h_start:j_h_end])] = [j_h_start, j_h_end]
                        if j_t_start >= t_1 and j_t_end <= t_2:
                            tentits[" ".join(doc[j_t_start:j_t_end])] = [j_t_start, j_t_end]
            bridge_ens_ids = set(hentits.keys()) & set(tentits.keys())

            bridge_unused = 5

            hbs_list = dict()
            tbs_list = dict()
            for bid in bridge_ens_ids:
                tmpbid = hentits[bid]
                s = tmpbid[0]
                t = tmpbid[1]
                hbs = [f'[unused{bridge_unused}]'] + doc[s:t] + [f'[unused{bridge_unused + 1}]']
                tbs = [f'[unused{bridge_unused}]'] + doc[s:t] + [f'[unused{bridge_unused + 1}]']
                hbs_list[str(s)] = hbs
                bridge_unused += 2
                tbs_list[str(s)] = tbs
            for hid in hentits:
                if hid not in bridge_ens_ids:
                    tmpbid = hentits[hid]
                    s = tmpbid[0]
                    t = tmpbid[1]
                    hbs = [f'*'] + doc[s:t] + [f'*']
                    hbs_list[str(s)] = hbs
            for tid in tentits:
                if tid not in bridge_ens_ids:
                    tmpbid = tentits[tid]
                    s = tmpbid[0]
                    t = tmpbid[1]
                    tbs = [f'*'] + doc[s:t] + [f'*']
                    tbs_list[str(s)] = tbs

            hbs_list[str(h_start)] = ['[unused1]'] + doc[h_start:h_end] + ['[unused2]']
            tbs_list[str(t_start)] = ['[unused3]'] + doc[t_start:t_end] + ['[unused4]']
            tmph_idx = h_1
            tmpt_idx = t_1
            h_tokens = []
            t_tokens = []
            while tmph_idx < h_2:
                if str(tmph_idx) in hbs_list.keys():
                    s = hbs_list[str(tmph_idx)]
                    h_tokens.extend(s)
                    tmph_idx += len(s) - 2
                else:
                    h_tokens.append(doc[tmph_idx])
                    tmph_idx += 1
            while tmpt_idx < t_2:
                if str(tmpt_idx) in tbs_list.keys():
                    s = tbs_list[str(tmpt_idx)]
                    t_tokens.extend(s)
                    tmpt_idx += len(s) - 2
                else:
                    t_tokens.append(doc[tmpt_idx])
                    tmpt_idx += 1

            h_name = doc[h_start:h_end]
            t_name = doc[t_start:t_end]
            h_token_ids = tokenizer.convert_tokens_to_ids(h_tokens)
            t_token_ids = tokenizer.convert_tokens_to_ids(t_tokens)

            selected_ret = process(tokenizer, " ".join(doc[h_start:h_end]), " ".join(doc[t_start:t_end]), h_tokens,
                                   t_tokens)
            for s_blk in selected_ret:
                while (tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                    s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(
                        ".")
            input_ids = tokenizer.build_inputs_with_special_tokens(h_token_ids, t_token_ids)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(h_token_ids, t_token_ids)
            obj = tokenizer._pad({'input_ids': input_ids, 'token_type_ids': token_type_ids}, max_length=args.seq_len,
                                 padding_strategy='max_length')
            _input_ids.append(obj['input_ids'])
            _token_type_ids.append(obj['token_type_ids'])
            _attention_mask.append(obj['attention_mask'])
            _rs.append(r)
            selected_rets.append(selected_ret)
        dplabel_t = torch.tensor(_rs, dtype=torch.long)
        rs_t = None
        r = None
        selected_inputs = torch.zeros(4, len(examples), CAPACITY, dtype=torch.int64)
        for ex, buf in enumerate(selected_rets):
            returns = buf.export_01_turn(out=(selected_inputs[0, ex], selected_inputs[1, ex], selected_inputs[2, ex]))
            sents_docs.append(returns[-1])
        selected_ids = selected_inputs[0]
        selected_att_mask = selected_inputs[1]
        selected_token_type = selected_inputs[2]
        selected_labels = selected_inputs[3]

    return dplabel_t, rs_t, [
        r], selected_ids, selected_att_mask, selected_token_type, selected_labels, selected_rets, sents_docs


def collate_fn_infer(batch, args, relation2id, tokenizer, redisd, encoder, sbert_wk):
    # assert len(batch) == 1
    batch = batch[0]
    h, t = batch[0].split('#')
    rs = [relation2id[r] for r in batch[1]]
    dps = batch[2]
    dplabel = []
    selected_rets = list()
    for doc1, doc2, l in dps:

        collec_doc1_titles = [doc1 for doc1, _, _ in dps]
        collec_doc2_titles = [doc2 for _, doc2, _ in dps]
        doc_entities = get_doc_entities(h, t, tokenizer, redisd, args.no_additional_marker, args.mask_entity,
                                        collec_doc1_titles, collec_doc2_titles)
        dps_count = 0
        selected_ret = process_example_ReoS(h, t, doc1, doc2, tokenizer, args.seq_len, redisd,
                                            args.no_additional_marker, args.mask_entity, encoder, sbert_wk,
                                            doc_entities, dps_count)
        dplabel.append(relation2id[l])
        for s_blk in selected_ret:
            while (tokenizer.convert_tokens_to_ids("|") in s_blk.ids):
                s_blk.ids[s_blk.ids.index(tokenizer.convert_tokens_to_ids("|"))] = tokenizer.convert_tokens_to_ids(".")
        selected_rets.append(selected_ret)
    selected_inputs = torch.zeros(4, len(dps), CAPACITY, dtype=torch.int64)
    sents_docs = []
    dplabel_t = torch.tensor(dplabel, dtype=torch.int64)
    for dp, buf in enumerate(selected_rets):
        returns = buf.export_01_turn(out=(selected_inputs[0, dp], selected_inputs[1, dp], selected_inputs[2, dp]))
        sents_docs.append(returns[-1])
    selected_ids = selected_inputs[0]
    selected_att_mask = selected_inputs[1]
    selected_token_type = selected_inputs[2]
    selected_labels = selected_inputs[3]

    return h, rs, t, selected_ids, selected_att_mask, selected_token_type, selected_labels, selected_rets, sents_docs, dplabel_t


class GRN(nn.Module):
    def __init__(self, e_emb, e_hidden, g_hidden, dp=0.1, layer=3, agg='gate'):
        super(GRN, self).__init__()
        torch.manual_seed(12345)
        self.layer = layer
        self.dp = dp

        self.slstm = SGRU(e_emb, e_hidden, g_hidden)

        self.agg = agg

        self.gate3 = nn.Linear(2 * e_hidden, e_hidden)

        self.S = nn.Parameter(torch.tensor([0.]))

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    def forward(self, features, adjs, doc_emb, edge_features=None, edge_adjs=None):
        # def forward(self, features, adjs):
        ### [batch, wnum, e_hid]
        edge_h = None

        e_h = features.unsqueeze(0)

        e_hs = []
        wnum = e_h.shape[-2]
        # print("e_h", e_h.shape)
        adjs = adjs
        g_emb = doc_emb.unsqueeze(0)
        batch = 1
        e_hid = e_h.shape[-1]

        if edge_features is not None:
            edge_h = edge_features.unsqueeze(0)
            edgenum = wnum + edge_h.shape[-2]
            # print("edge_h", edge_h.shape)
            # print("edge_adjs",edge_adjs.shape)

        # print("e_h", e_h.shape,wnum,e_hid)
        for i in range(self.layer):
            # 1.aggregation
            # s_neigh_s_h = self.mean(s_h, s2smatrix)
            # s_neigh_s_h = self.sum(s_h, s2smatrix)

            # B S E H
            if self.agg == 'gate':
                # 实体间的gate
                e_h_expand = e_h.unsqueeze(-2).expand(batch, wnum, wnum,
                                                      e_hid)  # [[[A],[A],[A]],[[B],[B],[B]],[[C],[C],[C]]]

                e_neigh_h = e_h_expand.transpose(-2, -3)  # [[[A],[B],[C]],[[A],[B],[C]],[[A],[B],[C]]]
                e_neigh_e_h = torch.cat((e_h_expand, e_neigh_h),
                                        dim=-1)  # [[[AA],[AB],[AC]],[[BA],[BB],[BC]],[[CA][CB][CC]]]
                es = self.gate3(e_neigh_e_h)  # 将上述扩展维度后的Batch×Entity×Entity经转化成gate的输入

                g3 = torch.sigmoid(es)  # 即lattice论文的公式
                # two layer
                e_neigh_e_h = e_h_expand * adjs.unsqueeze(
                    -1) * g3  # [[[A],[B],[C]],[[A],[B],[C]],[[A],[B],[C]]]与mask和gate元素积

                e_neigh_e_h = e_neigh_e_h.sum(2)  # 维度2（第3级维度）上进行加和，变回Batch×Entity×隐向量维度

            e_input = torch.cat((e_h, e_neigh_e_h), -1)
            # print("e_input",e_input.shape)

            # 2.update entity embedding
            e_h, g_emb = self.slstm(e_input, e_h, g_emb)
            # print("e_h_after_slstm", e_h.shape)

            if edge_features is not None:
                tmp_edgeh = torch.cat((edge_h, e_h), dim=1)
                # print("tmp_edgeh",tmp_edgeh.shape)
                tmp_edgeh_expand = tmp_edgeh.unsqueeze(-2).expand(batch, edgenum, edgenum,
                                                                  e_hid)  # [[[A],[A],[A]],[[B],[B],[B]],[[C],[C],[C]]]

                tmp_edge_neigh_h = tmp_edgeh_expand.transpose(-2, -3)  # [[[A],[B],[C]],[[A],[B],[C]],[[A],[B],[C]]]
                # tmp_sh_neigh_e_h = torch.cat((tmp_sh_expand, tmp_sh_neigh_h),
                #                         dim=-1)  # [[[AA],[AB],[AC]],[[BA],[BB],[BC]],[[CA][CB][CC]]]
                #
                es = tmp_edge_neigh_h * edge_adjs.unsqueeze(-1)
                edge_input = torch.mean(es, dim=1)[:, :len(edge_features), :]

                # print("edge_input", edge_input.shape)

                edge_input = torch.cat((edge_h, edge_input), -1)

                edge_h, _ = self.slstm(edge_input, edge_h, None)

        if self.dp > 0:
            e_h = F.dropout(e_h, self.dp, self.training)
            e_hs.append(e_h)
            if edge_features is not None:
                edge_h = F.dropout(edge_h, self.dp, self.training)
        return e_h, edge_h, e_hs


class Codred(torch.nn.Module):
    def __init__(self, args, num_relations):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        self.GRN = GRN(768, 768, 768)
        self.predictor_gcn = torch.nn.Linear(768 * 2 + 768, num_relations)
        self.predictor = torch.nn.Linear(self.bert.config.hidden_size, num_relations)
        ###used for y_rela
        self.predictor_balanced = torch.nn.Linear(self.bert.config.hidden_size, num_relations)

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)

        self.sen_bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sen_bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        weight = torch.ones(num_relations, dtype=torch.float32)
        weight[0] = 0.1
        self.d_model = 768
        self.reduced_dim = 256
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.aggregator = args.aggregator
        self.no_doc_pair_supervision = args.no_doc_pair_supervision
        self.matt = MatTransformer(h=8, d_model=self.d_model, hidden_size=1024, num_layers=2, device=torch.device(0))
        self.wu = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.wi = nn.Linear(self.d_model, self.d_model)
        self.ln1 = nn.Linear(self.d_model, self.d_model)
        self.gamma = 2
        self.alpha = 0.25
        self.beta = 0.01
        self.d_k = 64
        self.num_relations = num_relations
        self.ent_emb = nn.Parameter(torch.zeros(2, self.d_model))
        self.row_emb = nn.Parameter(torch.zeros(1000))
        self.col_emb = nn.Parameter(torch.zeros(1000))
        self.reset_params()

    def reset_params(self):
        nn.init.normal_(self.row_emb.data, 0, 1e-4)
        nn.init.normal_(self.col_emb.data, 0, 1e-4)

    def get_unbridge_en_idxs(self, input_ids):
        b_spans = [i[0] for i in (input_ids == 115).nonzero().detach().tolist()]

        b_span_ = []
        if len(b_spans) > 0 and len(b_spans) % 2 == 0:
            b_span_chunks = [b_spans[i:i + 2] for i in range(0, len(b_spans), 2)]

        elif len(b_spans) > 0 and len(b_spans) % 2 == 1:
            b_span = []
            ptr = 0
            # pdb.set_trace()
            while (ptr <= len(b_spans) - 1):
                try:
                    if input_ids[b_spans[ptr + 1]] - input_ids[b_spans[ptr]] == 1:
                        b_span.append([b_spans[ptr], b_spans[ptr + 1]])
                        ptr += 2
                    else:
                        ptr += 1
                except IndexError as e:
                    ptr += 1
            for bs in b_span:
                b_span_.extend(bs)
                if len(b_span_) % 2 != 0:
                    # print(b_spans)
                    lll = 2
            b_span_chunks = [b_span_[i:i + 2] for i in range(0, len(b_span_), 2)]
        else:
            b_span_ = []
            b_span_chunks = []
        return b_span_chunks

    def get_bag_en_embs(self, bag_span_chunks, input_ids, embedding, bag_len, ner_embedding=None, sen_spans=None,
                        sen_docs=None):
        ###ner_embedding is used for calculating cosine similarity to judge for semantic-related edges
        b_embs_dp = []
        ner_b_embs_dp = []
        b_embs_key = []
        b_span_chunks_dict = {}
        ner_b_span_chunks_dict = {}
        unb_docs = {}
        for dp, b_span_chunks in enumerate(bag_span_chunks):
            for b_span in b_span_chunks:
                # print(b_span_chunks)
                # print(b_span)
                tmp = input_ids[dp, b_span[0]+1:b_span[1]].detach().tolist()
                # print(tmp)
                tmp = [str(i) for i in tmp]
                tmp = " ".join(tmp)
                # print("tmp_after", tmp)
                if tmp not in b_span_chunks_dict:
                    b_span_chunks_dict[tmp] = {}
                    ner_b_span_chunks_dict[tmp] = {}
                if dp not in b_span_chunks_dict[tmp]:
                    b_span_chunks_dict[tmp][dp] = []
                    ner_b_span_chunks_dict[tmp][dp] = []

                tmp_emb = embedding[dp, b_span[0]]
                b_span_chunks_dict[tmp][dp].append(tmp_emb)

                if ner_embedding is not None:
                    tmp_ner_emb = ner_embedding[dp, b_span[0]]
                    ner_b_span_chunks_dict[tmp][dp].append(tmp_ner_emb)

                if sen_spans is not None:
                    sen_spans_dp = sen_spans[dp]

                    s_idx = 0

                    while s_idx <= len(sen_spans_dp) - 2:
                        sen_start = sen_spans_dp[s_idx]
                        sen_end = sen_spans_dp[s_idx + 1]
                        if (s_idx // 2) < len(sen_docs[dp]):
                            sen_doc = sen_docs[dp][s_idx // 2]
                            # print("len(sen_spans)", len(sen_spans))
                            b_start = b_span[0]
                            b_end = b_span[1]
                            if b_start >= sen_start and b_end <= sen_end:
                                if tmp not in unb_docs:
                                    unb_docs[tmp] = {}
                                if dp not in unb_docs[tmp]:
                                    unb_docs[tmp][dp] = []
                                    unb_docs[tmp][dp].append(sen_doc)

                        s_idx += 2

        for key in b_span_chunks_dict:
            b_emb = []
            tmp_dps = []
            for dp in b_span_chunks_dict[key]:
                for data in b_span_chunks_dict[key][dp]:
                    b_emb.append(data)
                    if dp not in tmp_dps:
                        tmp_dps.append(dp)
            b_emb = torch.logsumexp(torch.stack(b_emb), dim=0)

            b_embs_dp.append(b_emb)
            b_embs_key.append((key, tmp_dps))

        if ner_embedding is not None:

            for key in ner_b_span_chunks_dict:
                ner_b_emb = []
                for dp in ner_b_span_chunks_dict[key]:
                    for data in ner_b_span_chunks_dict[key][dp]:
                        ner_b_emb.append(data)

                # ner_b_emb = torch.mean(torch.stack(ner_b_emb), dim=0)
                ner_b_emb = torch.logsumexp(torch.stack(ner_b_emb), dim=0)
                ner_b_embs_dp.append(ner_b_emb)

        return b_embs_dp, b_embs_key, unb_docs, ner_b_embs_dp

    def get_path_index(self, i, j, bridge_range, un_bridge_range):

        ###path非桥接实体之间
        path_noedge_count = 0
        path_edge_count = 0
        path_noedge_edge_count = 0

        last = bridge_range[-1]
        last_start = last[0]
        last_end = last[1]
        # print("last_start and end",last_start, last_end)
        i_idx = 0
        j_idx = 0
        if i <= last_end and j <= last_end:
            for r_idx, r in enumerate(bridge_range):
                s = r[0]
                e = r[1]
                if i >= s and i <= e:
                    i_idx = r_idx
                    break
            for r_idx, r in enumerate(bridge_range):
                s = r[0]
                e = r[1]
                if j >= s and j <= e:
                    j_idx = r_idx
                    break
            if i_idx == j_idx:
                path_edge_count += 1
                # print("i,j is bridge ens")
        elif i > last_end and j > last_end:
            for r_idx, r in enumerate(un_bridge_range):
                s = r[0]
                e = r[1]
                if i >= s and i <= e:
                    i_idx = r_idx
                    break
            for r_idx, r in enumerate(un_bridge_range):
                s = r[0]
                e = r[1]
                if j >= s and j <= e:
                    j_idx = r_idx
                    break
            if i_idx == j_idx:
                path_noedge_count += 1
                # print("i,j is unbridge ens")
        elif i > last_end and j <= last_end:
            for r_idx, r in enumerate(un_bridge_range):
                s = r[0]
                e = r[1]
                if i >= s and i <= e:
                    i_idx = r_idx
                    break
            for r_idx, r in enumerate(bridge_range):
                s = r[0]
                e = r[1]
                if j >= s and j <= e:
                    j_idx = r_idx
                    break
            if i_idx == j_idx:
                path_noedge_edge_count += 1
                # print("iorj is unbridge ens")
        elif i <= last_end and j > last_end:
            for r_idx, r in enumerate(un_bridge_range):
                s = r[0]
                e = r[1]
                if i >= s and i <= e:
                    i_idx = r_idx
                    break
            for r_idx, r in enumerate(bridge_range):
                s = r[0]
                e = r[1]
                if j >= s and j <= e:
                    j_idx = r_idx
                    break
            if i_idx == j_idx:
                path_noedge_edge_count += 1
                # print("iorj is unbridge ens")
        # print("path_noedge_count",path_noedge_count,path_edge_count,path_noedge_edge_count)
        return path_noedge_count, path_edge_count, path_noedge_edge_count, i_idx, j_idx
    

    def forward(self, input_ids, token_type_ids, attention_mask, dplabel=None, rs=None, train=True, sents_docs=None):
        # def forwrad(self, datas):
        # for batch in datas:

        bag_len, seq_len = input_ids.size()
        if not train:
            ###for debiasing, we add two input_ids for y_bias and y_rela
            total_inputids = [input_ids, input_ids.clone(), input_ids.clone()]
        else:
            total_inputids = [input_ids]
        total_htlogits = []
        total_nums = []
        
        for total_idx, input_ids in enumerate(total_inputids):
            #if total_idx == 0:
            bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        return_dict=False)
            embedding = bert_outputs[0]

            attention = bert_outputs[-1][-1]

            
            if rs is not None or not train:
                entity_mask, entity_span_list = self.get_htb(input_ids)


                h_embs = []
                t_embs = []
                b_embs = []
                unb_embs = []

                ner_b_embs = []
                ner_unb_embs = []

                b_keys = []
                unb_keys = []
                unb_docs = []
                features = []
                ner_features = []
                r_embeddings = []
                bag_b_span_chunks = []
                bag_unb_span_chunks = []
                bag_sen_spans = []
                bag_sen_docs = []

                for dp in range(0, bag_len):
                    b_embs_dp = []
                    b_embs_dp_ner = []
                    sen_embs = []

                    try:
                        h_span = entity_span_list[dp][0]
                        t_span = entity_span_list[dp][1]
                        b_span_chunks = entity_span_list[dp][2]

                        sen_spans = entity_span_list[dp][3]
                        sents_dp_docs = sents_docs[dp]
                        ###get unbridge entities idxs
                        unb_span_chunks = self.get_unbridge_en_idxs(input_ids[dp])

                        h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1] + 1], dim=0)[0]
                        t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1] + 1], dim=0)[0]


                        h_embs.append(h_emb)
                        t_embs.append(t_emb)

                        bag_b_span_chunks.append(b_span_chunks)
                        bag_unb_span_chunks.append(unb_span_chunks)

                        bag_sen_spans.append(sen_spans)
                        bag_sen_docs.append(sents_dp_docs)

                    except IndexError as e:
                        continue
                    
                    
                
                tmp_b_embs, tmp_b_keys, _, tmp_b_ner_embs = self.get_bag_en_embs(bag_b_span_chunks, input_ids, embedding, bag_len, embedding)
                
                tmp_unb_embs, tmp_unb_keys, tmp_unb_docs, tmp_unb_ner_embs = self.get_bag_en_embs(bag_unb_span_chunks,
                                                                                                  input_ids,
                                                                                                  embedding,
                                                                                                  bag_len,
                                                                                                  embedding,
                                                                                                  bag_sen_spans,
                                                                                                  bag_sen_docs)

                def get_en_attnscores(tokenizer, attention,input_ids, entity_span_list, bag_unb_span_chunks):
                #use last layer batch x head x 512 x 512
                    avg_attention = torch.mean(attention,dim=1)
                    
                    # word_importance = torch.sum(avg_attention, dim=1)
                    tokens = []
                    bag_b_attns = {}
                    def get_b_attns(b_span_chunks, h_attn, t_attn, bridge):
                        b_span_attns = list()
                        for b_span_chunk in b_span_chunks:
                            tmp_attn = torch.sum(h_attn[:,b_span_chunk[0]:b_span_chunk[1]+1],dim=(0,1))
                            tmp_attn2 = torch.sum(t_attn[:,b_span_chunk[0]:b_span_chunk[1]+1],dim=(0,1))

                            b_span_attns.append([b_span_chunk, tmp_attn.item(), tmp_attn2.item(), dp, bridge])
                        # b_span_attns = sorted(b_span_attns, key= lambda x: x[1]+x[2], reverse=True)
                        return b_span_attns
                    # Convert token IDs to tokens and map attention weights to tokens
                    total_attns = list()
                    for dp in range(input_ids.shape[0]):
                        h_span = entity_span_list[dp][0]
                        h_attn = avg_attention[dp,h_span]
                        t_span = entity_span_list[dp][1]
                        t_attn = avg_attention[dp,t_span]
                        b_span_chunks = entity_span_list[dp][2]
                        b_span_attns = get_b_attns(b_span_chunks, h_attn, t_attn, 0)
                        if dp < len(bag_unb_span_chunks):
                            unb_span_chunks = bag_unb_span_chunks[dp]
                            unb_span_attns = get_b_attns(unb_span_chunks, h_attn, t_attn, 1)
                        else:
                            unb_span_attns = []
                        all_span_attns = b_span_attns+unb_span_attns
                        all_span_attns = sorted(all_span_attns, key=lambda x:x[1]+x[2], reverse=True)
                        total_attns.extend(all_span_attns)
                        # unb_span_chunks = bag_unb_span_chunks[dp]
                        # token = self.tokenizer.convert_ids_to_tokens(input_ids[dp].tolist())
                        # tokens.append(token)
                        bag_b_attns[dp] = all_span_attns

                    # for batch_idx in range(word_importance.shape[0]):
                    #     token_importance_batch = list(zip(tokens[batch_idx], word_importance[batch_idx].tolist()))
                    #     # token_importance_batch.sort(key=lambda x: x[1], reverse=True)
                    #     token_importance.append(token_importance_batch)
                    total_attns = sorted(total_attns, key=lambda x:x[1]+x[2], reverse=True)
                    return bag_b_attns, total_attns

                
                b_embs = tmp_b_embs
                unb_embs = tmp_unb_embs

                ner_b_embs = tmp_b_ner_embs
                ner_unb_embs = tmp_unb_ner_embs

                b_keys = tmp_b_keys
                unb_keys = tmp_unb_keys

                unb_docs = tmp_unb_docs
                
                if not train:
                    dp_en_importance, bag_en_importance = get_en_attnscores(self.tokenizer, attention, input_ids, entity_span_list, bag_unb_span_chunks)
                    topk = len(bag_en_importance)
                    if total_idx==1:
                        ###for y_bias to select important non-tagret entities
                        topk = int(0.5*len(bag_en_importance))
                    if topk==0:
                        topk = len(bag_en_importance)
                    bag_en_importance = bag_en_importance[topk:]
                    b_attn_keys = []
                    unb_attn_keys = []
                    # pdb.set_trace()
                    for item in bag_en_importance:
                        e_start = item[0][0]
                        e_end = item[0][-1]

                        bridge = item[-1]
                        e_dp = item[-2]
                        e_range = input_ids[e_dp, e_start+1:e_end].detach().tolist()
                        e_range = [str(i) for i in e_range]
                        e_range = " ".join(e_range)
                        e_key_index = [0,0]
                        if bridge ==0:
                            for b_key_index,b_key_item in enumerate(b_keys):
                                if e_range == b_key_item[0] and e_dp in b_key_item[1]:
                                    e_key_index[0] = b_key_index
                                    e_key_index[1] = e_dp
                                    b_attn_keys.append(e_key_index) 
                                    break
                        elif bridge == 1:
                            for b_key_index,b_key_item in enumerate(unb_keys):
                                if e_range == b_key_item[0] and e_dp in b_key_item[1]:
                                    e_key_index[0] = b_key_index
                                    e_key_index[1] = e_dp
                                    unb_attn_keys.append(e_key_index)
                                    break
                if total_idx==1:
                    b_attn_keys_indexs = list()
                    for x in b_attn_keys:
                        if x[0] not in b_attn_keys_indexs:
                            b_attn_keys_indexs.append(x[0])
                    # b_attn_keys_indexs = [x[0] for x in b_attn_keys]
                    b_embs = [x for x_i, x in enumerate(b_embs) if x_i in b_attn_keys_indexs]
                    ner_b_embs = [x for x_i, x in enumerate(ner_b_embs) if x_i in b_attn_keys_indexs]

                    unb_attn_keys_indexs = list()
                    for x in unb_attn_keys:
                        if x[0] not in unb_attn_keys_indexs:
                            unb_attn_keys_indexs.append(x[0])
                    # unb_attn_keys_indexs = [x[0] for x in unb_attn_keys]
                    unb_embs = [x for x_i, x in enumerate(unb_embs) if x_i in unb_attn_keys_indexs]
                    ner_unb_embs = [x for x_i, x in enumerate(ner_unb_embs) if x_i in unb_attn_keys_indexs]
                    
                features.extend(b_embs)
                features.extend(unb_embs)
                ner_features.extend(ner_b_embs)
                ner_features.extend(ner_unb_embs)
                len_b_embs = len(b_embs)
                len_unb_embs = len(unb_embs)
                total_num = len(h_embs) + len(t_embs) + len_b_embs + len_unb_embs
                # pdb.set_trace()

                start_index = -1
                for i in range(len(h_embs)):
                    features.append(h_embs[start_index])
                    features.append(t_embs[start_index])
                    start_index = start_index - 1

                adjs = np.zeros((total_num, total_num))
                t_index = [-1 - (2 * t) for t in range(len(t_embs))]

                edge_index = []
                edge_features = []
                edge_count = 0

                ###Each bridge entity in a path is connected to the head and tail entities of its own path
                for en_index in range(len(b_embs)):
                    # path_h_idx = start_index-1
                    # path_t_idx = start_index
                    if total_idx==1:
                        if en_index in b_attn_keys_indexs:
                            tmp_en_index = b_attn_keys_indexs[en_index]
                        else:
                            tmp_en_index = en_index
                    else:
                        tmp_en_index = en_index

                    for dp_index in b_keys[tmp_en_index][1]:
                        path_h_idx = t_index[dp_index] - 1
                        path_t_idx = t_index[dp_index]

                        adjs[path_h_idx][en_index] = 1
                        adjs[en_index][path_h_idx] = 1

                        adjs[path_t_idx][en_index] = 1
                        adjs[en_index][path_t_idx] = 1

                ###Non-bridge entities connect to path head and tail entities
                for en_index in range(len(unb_embs)):
                    # tmp_en_index = en_index
                    if total_idx==1:
                        if en_index in unb_attn_keys_indexs:
                            tmp_en_index = unb_attn_keys_indexs[en_index]
                        else:
                            tmp_en_index = en_index
                    else:
                        tmp_en_index = en_index
                    if unb_keys[tmp_en_index][0] in unb_docs:
                        unb_doc = unb_docs[unb_keys[tmp_en_index][0]]
                        for dp_index in unb_keys[tmp_en_index][1]:
                            path_h_idx = t_index[dp_index] - 1
                            path_t_idx = t_index[dp_index]
                            if dp_index in unb_doc:
                                tmp_b_index = en_index + len(b_embs)
                                if unb_doc[dp_index][0] == 0:
                                    adjs[path_h_idx][tmp_b_index] = 1
                                    adjs[tmp_b_index][path_h_idx] = 1
                                elif unb_doc[dp_index][0] == 1:
                                    adjs[path_t_idx][tmp_b_index] = 1
                                    adjs[tmp_b_index][path_t_idx] = 1

                features = torch.stack(features, dim=0).cuda()

                edge_count = 0

                path_noedge_count = 0
                path_edge_count = 0
                path_noedge_edge_count = 0

                def cos_sim_2d(x, y):
                    norm_x = x / x.norm(dim=1)[:, None]
                    norm_y = y / y.norm(dim=1)[:, None]
                    return torch.matmul(norm_x, norm_y.transpose(0, 1))

                if len(ner_features) != 0:
                    ner_features = torch.stack(ner_features, dim=0).cuda()
                    ner_sim_matrix = cos_sim_2d(ner_features, ner_features)
                    total_count = 0
                    for i in range(len_b_embs + len_unb_embs - 1):

                        for j in range(i + 1, len_b_embs + len_unb_embs):
                            ner_sim = ner_sim_matrix[i][j].item()
                            if ner_sim >= 0.6:
                                # print("ner_sim", ner_sim, sim, i, j)
                                adjs[i][j] = 1
                                adjs[j][i] = 1

                adjs_torch = torch.from_numpy(adjs.copy()).type(torch.float32).cuda()
                doc_emb = torch.zeros_like(features[-1]).cuda()
                edge_features = None
                edge_adjs_torch = None
                output, edge_output, ehs = self.GRN(features, adjs_torch, doc_emb, edge_features, edge_adjs_torch)

                ###used for GRN
                output = output.squeeze(0)

                start_index = -1

                r_embeddings = []
                for i in range(len(h_embs)):
                    tail_hid = features[start_index]
                    head_hid = features[start_index - 1]
                    r_embeddings.extend([head_hid, tail_hid])
                    start_index = start_index - 2

                if total_idx==1 and len_b_embs==0:
                    edge_output = output[:len_unb_embs]
                else:
                    edge_output = output[:len_b_embs]
                r_embeddings.extend(edge_output)
                # print("len(r_embeddings)",len( r_embeddings))
                r_embeddings = torch.stack(r_embeddings, dim=0)

                u = self.wu(r_embeddings)
                v = self.wv(r_embeddings)
                alpha = u.view(1, r_embeddings.shape[0], 1, r_embeddings.shape[-1]) + v.view(1, 1,
                                                                                             r_embeddings.shape[0],
                                                                                             r_embeddings.shape[-1])
                # print("alpha",alpha.shape)
                # print("len(r_embeddings)",len( r_embeddings))
                alpha = F.relu(alpha)

                rel_enco = F.relu(self.ln1(alpha))

                r_mask = torch.ones(1, r_embeddings.shape[0], r_embeddings.shape[0]).cuda()
                # print("r_mask", r_mask.shape)
                rel_enco_m = self.matt(rel_enco, r_mask)
                j = 0
                rel_enco_m_ht = []
                while j < 2 * len(h_embs):
                    rel_enco_m_ht.append(rel_enco_m[0][j][j + 1])
                    j += 2
                t_feature_m = torch.stack(rel_enco_m_ht)
                if total_idx != 2:
                    ht_logits = self.predictor(t_feature_m)
                else:
                    ht_logits = self.predictor_balanced(t_feature_m)
                total_htlogits.append(ht_logits)
                # print("embeddings",embedding.shape)

                # ht_logits = self.predictor_gat(r_embeddings)
                bag_logit = torch.max(ht_logits, dim=0)[0]

            else:  # Inner doc

                entity_mask, entity_span_list = self.get_htb(input_ids)

                path_logits = []
                ht_logits_flatten_list = []
                for dp in range(0, bag_len):
                    h_embs = []
                    t_embs = []
                    b_embs = []
                    unb_embs = []

                    ner_b_embs = []
                    ner_unb_embs = []

                    b_keys = []
                    unb_keys = []
                    unb_docs = []
                    features = []
                    ner_features = []

                    dp_embs = []
                    r_embeddings = []

                    bag_b_span_chunks = []
                    bag_unb_span_chunks = []
                    bag_sen_spans = []
                    bag_sen_docs = []
                    # try:

                    h_span = entity_span_list[dp][0]
                    t_span = entity_span_list[dp][1]
                    b_span_chunks = entity_span_list[dp][2]

                    sen_spans = entity_span_list[dp][3]
                    sents_dp_docs = sents_docs[dp]

                    unb_span_chunks = self.get_unbridge_en_idxs(input_ids[dp])

                    h_emb = torch.max(embedding[dp, h_span[0]:h_span[-1] + 1], dim=0)[0]
                    t_emb = torch.max(embedding[dp, t_span[0]:t_span[-1] + 1], dim=0)[0]

                    # h_embs.append(h_emb)
                    # t_embs.append(t_emb)

                    h_embs.append(h_emb)
                    t_embs.append(t_emb)

                    bag_b_span_chunks.append(b_span_chunks)
                    bag_unb_span_chunks.append(unb_span_chunks)

                    bag_sen_spans.append(sen_spans)
                    bag_sen_docs.append(sents_dp_docs)

                    tmp_embedding = embedding[dp].unsqueeze(0)
                    # print("tmp_embedding",tmp_embedding.shape)
                    tmp_b_embs, tmp_b_keys, _, tmp_b_ner_embs = self.get_bag_en_embs(bag_b_span_chunks, input_ids,
                                                                                     tmp_embedding, bag_len,
                                                                                     embedding)

                    tmp_unb_embs, tmp_unb_keys, tmp_unb_docs, tmp_unb_ner_embs = self.get_bag_en_embs(
                        bag_unb_span_chunks,
                        input_ids,
                        tmp_embedding,
                        bag_len,
                        embedding,
                        bag_sen_spans,
                        bag_sen_docs)

                    b_embs = tmp_b_embs
                    unb_embs = tmp_unb_embs

                    ner_b_embs = tmp_b_ner_embs
                    ner_unb_embs = tmp_unb_ner_embs

                    b_keys = tmp_b_keys
                    unb_keys = tmp_unb_keys

                    unb_docs = tmp_unb_docs

                    len_b_embs = len(b_embs)
                    len_unb_embs = len(unb_embs)

                    total_num = len(h_embs) + len(t_embs) + len_b_embs + len_unb_embs
                    # total_num = len(h_embs) + len(t_embs) + len_b_embs
                    # print("len_b_embs len_unb_embs", len_b_embs, len_unb_embs)

                    features.extend(b_embs)

                    features.extend(unb_embs)

                    ner_features.extend(ner_b_embs)
                    ner_features.extend(ner_unb_embs)

                    start_index = -1
                    for i in range(len(h_embs)):
                        features.append(h_embs[start_index])
                        features.append(t_embs[start_index])
                        start_index = start_index - 1
                        
                    adjs = np.zeros((total_num, total_num))
                    # b_keys_start_index = [sum([len(b) for b in b_keys[:i]]) for i in range(len(b_keys))]
                    # unb_keys_start_index = [sum([len(b) for b in unb_keys[:i]]) + len_b_embs for i in range(len(unb_keys))]
                    t_index = [-1 - (2 * t) for t in range(len(t_embs))]

                    edge_index = []
                    edge_features = []
                    edge_count = 0
                    ###Each bridge entity in a path is connected to the head and tail entities of its own path
                    for en_index in range(len(b_embs)):
                        # path_h_idx = start_index-1
                        # path_t_idx = start_index
                        tmp_b_index = en_index

                        for dp_index in b_keys[en_index][1]:
                            path_h_idx = t_index[dp_index] - 1
                            path_t_idx = t_index[dp_index]

                            adjs[path_h_idx][tmp_b_index] = 1
                            adjs[tmp_b_index][path_h_idx] = 1

                            adjs[path_t_idx][tmp_b_index] = 1
                            adjs[tmp_b_index][path_t_idx] = 1

                    ###Non-bridge entities connect to path head and tail entities
                    for en_index in range(len(unb_embs)):
                        if unb_keys[en_index][0] in unb_docs:
                            unb_doc = unb_docs[unb_keys[en_index][0]]
                            for dp_index in unb_keys[en_index][1]:
                                path_h_idx = t_index[dp_index] - 1
                                path_t_idx = t_index[dp_index]
                                if dp_index in unb_doc:
                                    tmp_b_index = en_index + len(b_embs)
                                    if unb_doc[dp_index][0] == 0:
                                        # if unb_doc[tmp_unb_key][b_index] == 0:
                                        adjs[path_h_idx][tmp_b_index] = 1
                                        adjs[tmp_b_index][path_h_idx] = 1

                                        # edge_count += 1
                                        # edge_dp_index.append(dp_index)
                                    elif unb_doc[dp_index][0] == 1:
                                        # elif unb_doc[tmp_unb_key][b_index] == 1:
                                        adjs[path_t_idx][tmp_b_index] = 1
                                        adjs[tmp_b_index][path_t_idx] = 1

                    features = torch.stack(features, dim=0).cuda()

                    edge_count = 0

                    path_noedge_count = 0
                    path_edge_count = 0
                    path_noedge_edge_count = 0

                    def cos_sim_2d(x, y):
                        norm_x = x / x.norm(dim=1)[:, None]
                        norm_y = y / y.norm(dim=1)[:, None]
                        return torch.matmul(norm_x, norm_y.transpose(0, 1))

                    if len(ner_features) != 0:
                        ner_features = torch.stack(ner_features, dim=0).cuda()
                        ner_sim_matrix = cos_sim_2d(ner_features, ner_features)
                        total_count = 0
                        for i in range(len_b_embs + len_unb_embs - 1):
                            for j in range(i + 1, len_b_embs + len_unb_embs):
                                ner_sim = ner_sim_matrix[i][j].item()
                                if ner_sim >= 0.6:
                                    # print("ner_sim", ner_sim, sim, i, j)
                                    adjs[i][j] = 1
                                    adjs[j][i] = 1

                    adjs_torch = torch.from_numpy(adjs.copy()).type(torch.float32).cuda()

                    # edge_adjs_torch = torch.from_numpy(edge_adjs.copy()).type(torch.float32).cuda()

                    doc_emb = torch.zeros_like(features[-1]).cuda()

                    # ne = features.size()[0]
                    # features = features + self.row_emb[:ne].view(ne,1) + self.col_emb[:ne].view(ne,1)
                    # if len(tmp_edge_features) == 0:
                    edge_features = None
                    edge_adjs_torch = None
                    output, edge_output, ehs = self.GRN(features, adjs_torch, doc_emb, edge_features, edge_adjs_torch)

                    ###used for GRN
                    output = output.squeeze(0)

                    start_index = -1

                    r_embeddings = []
                    for i in range(len(h_embs)):
                        tail_hid = features[start_index]
                        head_hid = features[start_index - 1]
                        r_embeddings.extend([head_hid, tail_hid])
                        start_index = start_index - 2

                    edge_output = output[:len_b_embs]
                    r_embeddings.extend(edge_output)
                    r_embeddings = torch.stack(r_embeddings, dim=0)

                    u = self.wu(r_embeddings)
                    v = self.wv(r_embeddings)
                    alpha = u.view(1, r_embeddings.shape[0], 1, r_embeddings.shape[-1]) + v.view(1, 1,
                                                                                                 r_embeddings.shape[0],
                                                                                                 r_embeddings.shape[-1])
                    alpha = F.relu(alpha)

                    rel_enco = F.relu(self.ln1(alpha))

                    r_mask = torch.ones(1, r_embeddings.shape[0], r_embeddings.shape[0]).cuda()
                    # print("r_mask", r_mask.shape)
                    rel_enco_m = self.matt(rel_enco, r_mask)

                    t_feature = rel_enco_m
                    bs, es, es, d = rel_enco.size()
                    if total_idx!=2:
                        predict_logits = self.predictor(t_feature.reshape(bs, es, es, d))
                    else:
                        ###used for y_rela
                        predict_logits = self.predictor_balanced(t_feature.reshape(bs, es, es, d))

                    ht_logits = []

                    j = 0
                    while j < 2 * len(h_embs):
                        ht_logits.append(predict_logits[0][j, j + 1])
                        j += 2

                    ht_logits = torch.stack(ht_logits)
                    # print("ht_logits", ht_logits.shape)
                    tmpht_logits = ht_logits

                    _ht_logits_flatten = ht_logits.reshape(1, -1, self.num_relations)

                    ht_logits = tmpht_logits
                    path_logits.append(ht_logits)
                    ht_logits_flatten_list.append(_ht_logits_flatten)
                    # except Exception as e:
                    #     print(e)
                    #     pdb.set_trace()
                # pdb.set_trace()
                try:
                    path_logit = torch.stack(path_logits).reshape(1, 1, -1, self.num_relations).squeeze(0).squeeze(0)
                except Exception as e:
                    print(e)
                    pdb.set_trace()
        tmpflatten = []
        tmpfixedlow = []
        if dplabel is not None and rs is None and train:
            ### Inner doc
            ht_logits_flatten = torch.stack(ht_logits_flatten_list).squeeze(1)
            ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
            y_true = torch.zeros_like(ht_logits_flatten)
            for idx, dpl in enumerate(dplabel):
                y_true[idx, 0, dpl.item()] = 1
            bag_logit = path_logit
            loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low + 2, ht_fixed_low)

        elif rs is not None and train:
            _, prediction = torch.max(bag_logit.unsqueeze(0), dim=1)
            if self.no_doc_pair_supervision:
                pass
            else:
                ###ht_logits: [path, num_relations]
                ###ht_logits_flatten: [path, 1, num_relations]
                ht_logits_flatten = ht_logits.unsqueeze(1)
                y_true = torch.zeros_like(ht_logits_flatten)
                ###[:, :, 0]: relation=n/a
                ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
                # tmp_dplabel = [0]*(dp_logit.shape[0])

                if rs.item() != 0:

                    for idx, dpl in enumerate(dplabel):
                        try:
                            y_true[idx, :, dpl.item()] = torch.ones_like(y_true[idx, :, dpl.item()])
                            # tmp_dplabel[idx] = dpl.item()
                        except:
                            print("unmatched")
                # tmp_dplabel = torch.tensor(tmp_dplabel, dtype=torch.int64).cuda()
                # print('tmp_dplabel', tmp_dplabel.shape, dplabel.shape)
                #                 # pdb.set_trace()
                loss = self._multilabel_categorical_crossentropy(ht_logits_flatten, y_true, ht_fixed_low + 2,
                                                                 ht_fixed_low)
        
        else:
            ###not train
            for ht_logits in total_htlogits:
                ht_logits_flatten = ht_logits.unsqueeze(1)
                ht_fixed_low = (torch.ones_like(ht_logits_flatten) * 8)[:, :, 0].unsqueeze(-1)
                tmpflatten.append(ht_logits_flatten.transpose(0, 1))
                tmpfixedlow.append((ht_fixed_low + 2).transpose(0, 1))
            _, prediction = torch.max(bag_logit.unsqueeze(0), dim=1)

            loss = None
        if train:
            tmpflatten = ht_logits_flatten
            tmpfixedlow = ht_fixed_low
        prediction = []
        return loss, prediction, bag_logit, tmpflatten, tmpfixedlow, (bag_len)

    def _multilabel_categorical_crossentropy(self, y_pred, y_true, cr_ceil, cr_low, ghm=True, r_dropout=True):
        # cr_low + 2 = cr_ceil
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = torch.cat([y_pred_neg, cr_ceil], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, -cr_low], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return ((neg_loss + pos_loss + cr_low.squeeze(-1) - cr_ceil.squeeze(-1))).mean()

    def graph_encode(self, ent_encode, rel_encode, ent_mask, rel_mask):
        bs, ne, d = ent_encode.size()
        ent_encode = ent_encode + self.ent_emb[0].view(1, 1, d)
        rel_encode = rel_encode + self.ent_emb[1].view(1, 1, 1, d)
        rel_encode, ent_encode = self.graph_enc(rel_encode, ent_encode, rel_mask, ent_mask)
        return rel_encode

    def get_htb(self, input_ids):
        htb_mask_list = []
        htb_list_batch = []
        for pi in range(input_ids.size()[0]):
            # pdb.set_trace()
            tmp = torch.nonzero(input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
            if tmp.size()[0] < input_ids.size()[0]:
                # print(input_ids)
                lll = 2
            try:

                h_starts = [i[0] for i in (input_ids[pi] == H_START_MARKER_ID).nonzero().detach().tolist()]
                sen_starts = [i[0] for i in (input_ids[pi] == 101).nonzero().detach().tolist()]
                # sen_ends = [0] * len(sen_starts)
                sen_ends = [i[0] for i in (input_ids[pi] == 102).nonzero().detach().tolist()]

                # sen_ends[-1] = len(input_ids[pi]) - 1
                tmp_i = 0
                sen_span = []
                # for sen_start in sen_starts[1:]:
                #     sen_ends[tmp_i] = sen_start - 1
                #     tmp_i += 1
                for sen_start, sen_end in zip(sen_starts, sen_ends):
                    sen_span.append(sen_start)
                    sen_span.append(sen_end)
                # print("sen_span",sen_span)
                # print("sen_starts", sen_starts)
                # print("sen_ends", sen_ends)

                h_ends = [i[0] for i in (input_ids[pi] == H_END_MARKER_ID).nonzero().detach().tolist()]
                t_starts = [i[0] for i in (input_ids[pi] == T_START_MARKER_ID).nonzero().detach().tolist()]
                t_ends = [i[0] for i in (input_ids[pi] == T_END_MARKER_ID).nonzero().detach().tolist()]
                if len(h_starts) == len(h_ends):
                    h_start = h_starts[0]
                    h_end = h_ends[0]
                else:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                                break
                if len(t_starts) == len(t_ends):
                    t_start = t_starts[0]
                    t_end = t_ends[0]
                else:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e
                                break
                if h_end - h_start <= 0 or t_end - t_start <= 0:
                    # print(h_starts)
                    # print(h_ends)
                    # print(t_starts)
                    # print(t_ends)
                    # pdb.set_trace()
                    if h_end - h_start <= 0:
                        for h_s in h_starts:
                            for h_e in h_ends:
                                if 0 < h_e - h_s < 20:
                                    h_start = h_s
                                    h_end = h_e
                                    break
                    if t_end - t_start <= 0:
                        for t_s in t_starts:
                            for t_e in t_ends:
                                if 0 < t_e - t_s < 20:
                                    t_start = t_s
                                    t_end = t_e
                                    break
                    if h_end - h_start <= 0 or t_end - t_start <= 0:
                        pdb.set_trace()

                b_spans = torch.nonzero(
                    torch.gt(torch.full(([input_ids.size()[1]]), 99).to(input_ids.device), input_ids[pi])).squeeze(
                    0).squeeze(1).detach().tolist()

                token_len = input_ids[pi].nonzero().size()[0]
                b_spans = [i for i in b_spans if i <= token_len - 1]

                assert len(b_spans) >= 4
                # for i in [h_start, h_end, t_start, t_end]:
                for i in h_starts + h_ends + t_starts + t_ends:
                    b_spans.remove(i)
                # print("b_spans",b_spans)
                # sss
                h_span = [h_pos for h_pos in range(h_start, h_end + 1)]
                t_span = [t_pos for t_pos in range(t_start, t_end + 1)]
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(h_span).to(
                    input_ids.device), 1)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(t_span).to(
                    input_ids.device), 1)
            except:  # dps＜8
                # pdb.set_trace()
                h_span = []
                t_span = []
                h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
                t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
                b_spans = []

            b_span_ = []
            if len(b_spans) > 0 and len(b_spans) % 2 == 0:
                b_span_chunks = [b_spans[i:i + 2] for i in range(0, len(b_spans), 2)]
                b_span = []
                for span in b_span_chunks:
                    b_span.extend([b_pos for b_pos in range(span[0], span[1] + 1)])
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span).to(
                    input_ids.device), 1)
                b_span_.extend(b_span)
            elif len(b_spans) > 0 and len(b_spans) % 2 == 1:
                b_span = []
                ptr = 0
                # pdb.set_trace()
                while (ptr <= len(b_spans) - 1):
                    try:
                        if input_ids[pi][b_spans[ptr + 1]] - input_ids[pi][b_spans[ptr]] == 1:
                            b_span.append([b_spans[ptr], b_spans[ptr + 1]])
                            ptr += 2
                        else:
                            ptr += 1
                    except IndexError as e:
                        ptr += 1
                for bs in b_span:
                    b_span_.extend(bs)
                    if len(b_span_) % 2 != 0:
                        # print(b_spans)
                        lll = 2
                b_span_chunks = [b_span_[i:i + 2] for i in range(0, len(b_span_), 2)]
                b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(0, torch.tensor(b_span_).to(
                    input_ids.device), 1)
            else:
                b_span_ = []
                b_span_chunks = []
                b_mask = torch.zeros_like(input_ids[pi])
            htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(0), b_mask.unsqueeze(0)], dim=0)
            htb_mask_list.append(htb_mask)
            htb_list_batch.append([h_span, t_span, b_span_chunks, sen_span])
        htb_mask_batch = torch.stack(htb_mask_list, dim=0)
        return htb_mask_batch, htb_list_batch


def get_doc_entities(h, t, tokenizer, redisd, no_additional_marker, mask_entity, collec_doc1_titles,
                     collec_doc2_titles):
    max_len = 99999
    bert_max_len = 512
    Doc1_tokens = []
    Doc2_tokens = []
    B_entities = []
    for doc1_title, doc2_title in zip(collec_doc1_titles, collec_doc2_titles):
        doc1 = json.loads(redisd.get('codred-doc-' + doc1_title))
        doc2 = json.loads(redisd.get('codred-doc-' + doc2_title))
        v_h = None
        for entity in doc1['entities']:
            if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
                v_h = entity
        assert v_h is not None
        v_t = None
        for entity in doc2['entities']:
            if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
                v_t = entity
        assert v_t is not None
        d1_v = dict()
        for entity in doc1['entities']:
            if 'Q' in entity:
                d1_v[entity['Q']] = entity
        d2_v = dict()
        for entity in doc2['entities']:
            if 'Q' in entity:
                d2_v[entity['Q']] = entity
        ov = set(d1_v.keys()) & set(d2_v.keys())
        if len(ov) > 40:
            ov = set(random.choices(list(ov), k=40))
        ov = list(ov)
        ma = dict()
        for e in ov:
            ma[e] = len(ma)
        B_entities.append(ma)
    return B_entities


class CodredCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_argument(self, parser):
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--aggregator', type=str, default='attention')
        parser.add_argument('--positive_only', action='store_true')
        parser.add_argument('--positive_ep_only', action='store_true')
        parser.add_argument('--no_doc_pair_supervision', action='store_true')
        parser.add_argument('--no_additional_marker', action='store_true')
        parser.add_argument('--mask_entity', action='store_true')
        parser.add_argument('--single_path', action='store_true')
        parser.add_argument('--dsre_only', action='store_true')
        parser.add_argument('--raw_only', action='store_true')
        parser.add_argument('--load_model_path', type=str, default=None)
        parser.add_argument('--train_file', type=str, default='../data/rawdata/train_dataset.json')
        parser.add_argument('--dev_file', type=str, default='../data/rawdata/dev_dataset.json')
        parser.add_argument('--test_file', type=str, default='../data/rawdata/test_dataset.json')
        parser.add_argument('--dsre_file', type=str, default='../data/dsre_train_examples.json')
        parser.add_argument('--model_name', type=str, default='bert')

    def load_model(self):
        relations = json.load(open('../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index
        with self.trainer.cache():
            reasoner = Codred(self.args, len(self.relations))
            if self.args.load_model_path:
                load_model(reasoner, self.args.load_model_path)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        self.sbert_wk = sbert(device='cuda')
        return reasoner

    def load_data(self):
        train_dataset = json.load(open(self.args.train_file))
        dev_dataset = json.load(open(self.args.dev_file))
        test_dataset = json.load(open(self.args.test_file))
        if self.args.positive_only:
            train_dataset = [d for d in train_dataset if d[3] != 'n/a']
            dev_dataset = [d for d in dev_dataset if d[3] != 'n/a']
            # test_dataset = [d for d in test_dataset if d[3] != 'n/a']
        train_bags = place_train_data(train_dataset)
        dev_bags = place_dev_data(dev_dataset, self.args.single_path)
        test_bags = place_test_data(test_dataset, self.args.single_path)
        if self.args.positive_ep_only:
            train_bags = [b for b in train_bags if b[1] != 'n/a']
            dev_bags = [b for b in dev_bags if 'n/a' not in b[1]]
            test_bags = [b for b in test_bags if 'n/a' not in b[1]]
        self.dsre_train_dataset = json.load(open(self.args.dsre_file))
        self.dsre_train_dataset = [d for i, d in enumerate(self.dsre_train_dataset) if i % 10 == 0]

        d = list()
        for i in range(len(self.dsre_train_dataset) // 8):
            d.append(self.dsre_train_dataset[8 * i:8 * i + 8])
        d = random.sample(d, len(train_bags))
        if self.args.raw_only:
            pass
        elif self.args.dsre_only:
            train_bags = d
        else:
            d.extend(train_bags)
            train_bags = d
        self.redisd = redis.Redis(host='localhost', port=6379, decode_responses=True, db=0)

        with self.trainer.once():
            self.train_logger = Logger(['train_loss', 'train_acc', 'train_pos_acc', 'train_dsre_acc'],
                                       self.trainer.writer, self.args.logging_steps, self.args.local_rank)
            self.dev_logger = Logger(['dev_mean_prec', 'dev_f1', 'dev_auc'], self.trainer.writer, 1,
                                     self.args.local_rank)
            self.test_logger = Logger(['test_mean_prec', 'test_f1', 'test_auc'], self.trainer.writer, 1,
                                      self.args.local_rank)
        return train_bags, dev_bags, test_bags

    def collate_fn(self):
        return partial(collate_fn, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer,
                       redisd=self.redisd, encoder=self.bert, sbert_wk=self.sbert_wk), partial(collate_fn_infer,
                                                                                               args=self.args,
                                                                                               relation2id=self.relation2id,
                                                                                               tokenizer=self.tokenizer,
                                                                                               redisd=self.redisd,
                                                                                               encoder=self.bert,
                                                                                               sbert_wk=self.sbert_wk), partial(
            collate_fn_infer, args=self.args, relation2id=self.relation2id, tokenizer=self.tokenizer,
            redisd=self.redisd, encoder=self.bert, sbert_wk=self.sbert_wk)

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        with self.trainer.once():
            self.train_logger.log(train_loss=loss)
            if inputs['rs'] is not None:
                _, prediction, logit, ht_logits_flatten, ht_threshold_flatten, _ = outputs
                rs = extra['rs']
                if ht_logits_flatten is not None:
                    r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)

                    if r_score > ht_threshold_flatten[0, 0, 0]:
                        prediction = [r_idx.item()]
                    else:
                        prediction = [0]

                for p, score, gold in zip(prediction, logit, rs):
                    self.train_logger.log(train_acc=1 if p == gold else 0)
                    if gold > 0:
                        self.train_logger.log(train_pos_acc=1 if p == gold else 0)
            else:
                _, prediction, logit, ht_logits_flatten, ht_threshold_flatten, _ = outputs
                dplabel = inputs['dplabel']
                logit, dplabel = tensor_to_obj(logit, dplabel)
                prediction = []
                if ht_logits_flatten is not None:
                    r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)
                    for dp_i, (r_s, r_i) in enumerate(zip(r_score, r_idx)):
                        if r_s > ht_threshold_flatten[dp_i, 0, 0]:
                            prediction.append(r_i.item())
                        else:
                            prediction.append(0)
                for p, l in zip(prediction, dplabel):
                    self.train_logger.log(train_dsre_acc=1 if p == l else 0)
           

    def on_train_epoch_end(self, epoch):
        # print(epoch, self.train_logger.d)
        pass

    def on_dev_epoch_start(self, epoch):
        self._prediction = list()

    def on_dev_step(self, step, inputs, extra, outputs):
        _, prediction, logit, tmpht_logits_flatten, tmpht_threshold_flatten,_ = outputs
        ht_logits_flatten, enmaskpart_ht_logits_flatten, enenhance_ht_logits_flatten = tmpht_logits_flatten
        ht_threshold_flatten,  enmaskpart_ht_threshold_flatten, enenhance_ht_threshold_flatten = tmpht_threshold_flatten
        eval_logit = torch.max(ht_logits_flatten, dim=1)[0]
        ###for y_bias
        enmaskpart_eval_logit = torch.max(enmaskpart_ht_logits_flatten, dim=1)[0]
        ###for y_rela
        enenhance_eval_logit = torch.max(enenhance_ht_logits_flatten, dim=1)[0]
        
        h, t, rs = extra['h'], extra['t'], extra['rs']
        r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)


        # tmp_prediction = [r_idx.item()]
        if r_score > ht_threshold_flatten[:, 0, 0]:
            org_prediction = [r_idx.item()]
        else:
            org_prediction = [0]


        enhance_topk_idx = torch.topk(enenhance_eval_logit[0], 1).indices.tolist()[0]
        sub_topk_idx = torch.topk(enmaskpart_eval_logit[0], 1).indices.tolist()[0]
        org_eval_logit = eval_logit
        ###y = y  + 0.1*(y_rela - y_bias)
        eval_logit_addenhance = torch.tensor(eval_logit)
        eval_logit_addenhance[:,enhance_topk_idx] = eval_logit_addenhance[:,enhance_topk_idx] + 0.1*enenhance_eval_logit[:,enhance_topk_idx]
        eval_logit_addenhance[:,sub_topk_idx] = eval_logit_addenhance[:,sub_topk_idx] - 0.1*enmaskpart_eval_logit[:,sub_topk_idx]

        prediction = [r_idx.item()]
        eval_logit, logit = tensor_to_obj(eval_logit, logit)
        org_eval_logit = tensor_to_obj(org_eval_logit)
        
        tmpx = [eval_logit_addenhance[0]]
        self._prediction.append([org_prediction[0], prediction[0], eval_logit[0], tmpx,h, t, rs])

    def on_dev_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        pred_result = list()
        results = list()
        pred_result_enhance = list()
        facts = dict()

        for orgp, p, score, tmpscores,h, t, rs in self._prediction:
            tmprs = [r for r in rs]
            # print("p, tmprs", p, tmprs)
            rs = [self.relations[r] for r in rs]
            enhance_score = tmpscores[0]
            
            for i in range(1, len(score)):
                pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
            for i in range(1, len(enhance_score)):
                pred_result_enhance.append({'entpair': [h, t], 'relation': self.relations[i], 'score': enhance_score[i]})
            results.append([h, rs, t, self.relations[p]])
            for r in rs:
                if r != 'n/a':
                    facts[(h, t, r)] = 1
        stat = eval_performance(facts, pred_result)
        stat_enhance = eval_performance(facts, pred_result_enhance)
        #stat_enhance_label = eval_performance(facts, pred_result_enhance_label)                
        
        with self.trainer.once():
            json.dump(stat, open(f'output/dev-stat-dual-K1-{epoch}.json', 'w'))
            json.dump(stat_enhance, open(f'output/dev-enhance_0.1_sub0.1-{epoch}.json', 'w'))
            json.dump(results, open(f'output/dev-results-dual-K1-{epoch}.json', 'w'))
        return stat['f1']

    def on_test_epoch_start(self, epoch):
        self._prediction = list()
        pass
    def on_test_step(self, step, inputs, extra, outputs):
        _, prediction, logit, tmpht_logits_flatten, tmpht_threshold_flatten,_ = outputs
        # r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)
        # pdb.set_trace()
        ht_logits_flatten, enmaskpart_ht_logits_flatten, enenhance_ht_logits_flatten = tmpht_logits_flatten
        ht_threshold_flatten, enmaskpart_ht_threshold_flatten, enenhance_ht_threshold_flatten = tmpht_threshold_flatten
        eval_logit = torch.max(ht_logits_flatten, dim=1)[0]

        enmaskpart_eval_logit = torch.max(enmaskpart_ht_logits_flatten, dim=1)[0]
        enenhance_eval_logit = torch.max(enenhance_ht_logits_flatten, dim=1)[0]
        
        h, t, rs = extra['h'], extra['t'], extra['rs']
        r_score, r_idx = torch.max(torch.max(ht_logits_flatten, dim=1)[0], dim=-1)


        # tmp_prediction = [r_idx.item()]
        if r_score > ht_threshold_flatten[:, 0, 0]:
            org_prediction = [r_idx.item()]
            # eval_logit = eval_logit - 0.7 * enmask_eval_logit
        else:
            org_prediction = [0]

        topk_idx = torch.topk(eval_logit[0], 1).indices.tolist()
        enhance_topk_value = torch.topk(enenhance_eval_logit[0], 1).values.tolist()[0]
        enhance_topk_idx = torch.topk(enenhance_eval_logit[0], 1).indices.tolist()[0]
        sub_topk_idx = torch.topk(enmaskpart_eval_logit[0], 1).indices.tolist()[0]
        org_eval_logit = eval_logit
        
        eval_logit_addenhance = torch.tensor(eval_logit)
        eval_logit_addenhance[:,enhance_topk_idx] = eval_logit_addenhance[:,enhance_topk_idx] + 0.1*enenhance_eval_logit[:,enhance_topk_idx]
        eval_logit_addenhance[:,sub_topk_idx] = eval_logit_addenhance[:,sub_topk_idx] - 0.1*enmaskpart_eval_logit[:,sub_topk_idx]
    
        prediction = [r_idx.item()]
        eval_logit, logit = tensor_to_obj(eval_logit, logit)
        org_eval_logit = tensor_to_obj(org_eval_logit)
        
        tmpx = [eval_logit_addenhance[0]]
        self._prediction.append([org_prediction[0], prediction[0], eval_logit[0], tmpx,h, t, rs,])

    def on_test_epoch_end(self, epoch):
        self._prediction = self.trainer.distributed_broadcast(self._prediction)
        coda_file = dict()
        coda_file['setting'] = 'closed'

        nonacoda_file = dict()
        nonacoda_file['setting'] = 'closed'

        org_facts = dict()
        org_out_results = list()
        org_coda_file = dict()
        org_coda_file['setting'] = 'closed'
        org_coda_file1 = dict()
        org_coda_file1['setting'] = 'closed'


        enhance_01_results = list()
        enhance_01_coda_file = dict()
        enhance_01_coda_file['setting'] = 'closed'
        for orgp, p, score, tmpscores, h, t, rs in self._prediction:
            rs = [self.relations[r] for r in rs]
            score1 = tmpscores[0]
            
            for i in range(1, len(score)):
                #org_pred_result.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score[i]})
                org_out_results.append(
                    {'h_id': str(h), "t_id": str(t), "relation": str(self.relations[i]), "score": float(score[i])})
            
            for i in range(1, len(score1)):
                #org_pred_result1.append({'entpair': [h, t], 'relation': self.relations[i], 'score': score1[i]})
                enhance_01_results.append(
                    {'h_id': str(h), "t_id": str(t), "relation": str(self.relations[i]), "score": float(score1[i])})
            for r in rs:
                if r != 'n/a':
                    org_facts[(h, t, r)] = 1

        org_coda_file['predictions'] = org_out_results
        enhance_01_coda_file['predictions'] = enhance_01_results
        with self.trainer.once():
            json.dump(enhance_01_coda_file, open(f'output/test-enhance-results-{epoch}.json', 'w'))
            json.dump(org_coda_file, open(f'output/test-orgcodalab-results-{epoch}.json', 'w'))
        return True
    def process_train_data(self, data):
        # selected_rets = []
        # for d in data:
        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'rs': data[1],
            'dplabel': data[0],
            'train': True,
            'sents_docs': data[-1]
        }
        # selected_rets.append(selected_inputs)
        # return {'rs': data[0][2]}, selected_rets, {'selected_rets': data[0][7]}
        return {'rs': data[2]}, selected_inputs, {'selected_rets': data[7]}

    def process_dev_data(self, data):
        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'rs': data[1],
            'dplabel': data[-1],
            'train': False,
            'sents_docs': data[-2]
        }
        return {'h': data[0], 'rs': data[1], 't': data[2]}, selected_inputs, {'selected_rets': data[7]}

    def process_test_data(self, data):

        selected_inputs = {
            'input_ids': data[3],
            'attention_mask': data[4],
            'token_type_ids': data[5],
            'train': False,
            'sents_docs': data[-2]
        }
        return {'h': data[0], 'rs': data[1], 't': data[2]}, selected_inputs, {'selected_rets': data[7]}


def main():
    trainer = Trainer(CodredCallback())
    trainer.run()


if __name__ == '__main__':
    main()