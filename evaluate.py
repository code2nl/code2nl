from __future__ import division
import os
import argparse
import torch
import codecs
import glob
from nlgeval import NLGEval
import table
import table.IO
import opts
import warnings
import random
from collections import defaultdict
warnings.simplefilter("ignore")
parser = argparse.ArgumentParser(description='evaluate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.json'.format(opt.split))
opt.pre_word_vecs = os.path.join(opt.root_dir, opt.dataset, 'embedding')

if opt.beam_size > 0:
    opt.batch_size = 1

def com_score(ref, pre):
    r_list = []
    r_list.append(ref)
    nlgeval = NLGEval() 
    metrics_dict = nlgeval.compute_metrics(r_list, pre)
    return metrics_dict
def effect_len(js_list, r_list):
    # len_dict = defaultdict(int)
    ref_list = defaultdict(list)
    pre_list = defaultdict(list)
    for i in range(len(js_list)):
        l = len(js_list[i]['token'])
        id = int(l/3) 
        if id <= 7:
            ref_list[id].append(js_list[i])
            pre_list[id].append(r_list[i])
    return ref_list, pre_list

def com_metric(js_list, r_list):
    tgt_pre_list = []
    tgt_ref = []
    for pred, gold in zip(r_list, js_list):
        tgt_pre_list.append(' '.join(pred.tgt))
        tgt_ref.append(' '.join(gold['tgt']))
    metric = com_score(tgt_ref, tgt_pre_list)
  
    
    lay_pre_list = []
    lay_ref = []
    for pred, gold in zip(r_list, js_list):
        lay_pre_list.append(' '.join(pred.lay))
        lay_ref.append(' '.join(gold['lay']))
    lay_metric = com_score(lay_ref, lay_pre_list)

    print("tgt metric: {}".format(metric))
    print("lay metric: {}".format(lay_metric))
    for k in range(3):
    # k = random.randint(1,10)
        print("the {} example".format(k))

        print("Token {}".format(js_list[k]['src']))
        print("Gold layer: {}".format(js_list[k]['lay']))
        print("Pre layer: {}".format(r_list[k].lay))
        print("Gold tgt : {}".format(js_list[k]['tgt']))
        print("Pre tgt : {}".format(r_list[k].tgt))
        print(" ")
    return metric, lay_metric

def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = table.IO.read_anno_json(opt.anno, opt)
    # metric_name_list = ['tgt']
    prev_best = (None, None)
    # print(opt.model_path)
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        with torch.no_grad():
            translator = table.Translator(opt, dummy_opt.__dict__)
            data = table.IO.TableDataset(
                js_list, translator.fields, 0, None, False)
            test_data = table.IO.OrderedIterator(
                dataset=data, device=opt.gpu, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)
            # inference
            r_list = []
            for batch in test_data:
                r = translator.translate(batch)
                r_list += r
        
        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))

        metric, _ = com_metric(js_list, r_list)
    if opt.split == 'test':
        ref_dic, pre_dict = effect_len(js_list, r_list)
        for i in range(len(ref_dic)):
            js_list = ref_dic[i]
            r_list = pre_dict[i]
            print("the effect of length {}".format(i))
            metric, _ = com_metric(js_list, r_list)

        if prev_best[0] is None or float(metric['Bleu_1']) > prev_best[1]:
            prev_best = (fn_model, metric['Bleu_1'])

    if (opt.split == 'dev') and (prev_best[0] is not None):
        with codecs.open(os.path.join(opt.root_dir, opt.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
