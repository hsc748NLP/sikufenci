'''
Author: your name
Date: 2021-04-17 20:52:44
LastEditTime: 2021-04-18 12:27:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /BERT-NER/task.py
'''


from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
from boto3.docs import generate_docs
from botocore.vendored.six import with_metaclass
import re 
import numpy as np
from requests.api import get
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils import data
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def readfile(filename):#从外部读取文件
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    # print(data)
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")
    # def get_dev(self,data_dir):
    #     data=self._read_tsv((os.path.join(data_dir, "valid.txt"))
    #     return data
        


    def get_input(self,input):
        input_list=[]
        o_list=[]
        data=[]
        for i in input:
            input_list.append(i)
        o_list=['E' for i in range(0,len(input))]
        data.append((input_list,o_list))
        print(data)
        return data
    
    


    def create_dev(self,data,outputpath):
        with open(outputpath+'valid.txt','w') as fp:
            for i in data:
                for j in range(0,len(i[0])):
                    fp.write(str(i[0][j])+' '+str(i[1][j])+'\n')
            # fp.write('\n')

    # def get_input_examples(self,outputpath):
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
    
    def get_labels(self):
        return ["X","O",'B','I','E','S', "[CLS]", "[SEP]"]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            # print(text_a)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples


def get_test(inputpath,datapath):
        f=open(inputpath,'r',encoding='utf-8')
        test_list=f.read().splitlines()
        test=[]
        for j in test_list:
            j=re.sub(' ','',j)
            if len(j)<509:
               test.append(j)
            else:
               test.extend(clip_list(j,509))
        with open(datapath+'/test.txt','w',encoding='utf-8') as fp:
            for i in test:
                for j in i:
                    fp.write(str(j)+' '+'S'+'\n')
                fp.write('\n')

def get_train(inputpath,datapath):
        f=open(datapath+'/test.txt','r',encoding='utf-8')
        content=f.read()
        with open(datapath+'/train.txt','w',encoding='utf-8') as fp:
            fp.write(content)

    
def get_dev(inputpath,datapath):
        f=open(datapath+'/test.txt','r',encoding='utf-8')
        content=f.read()
        with open(datapath+'/valid.txt','w',encoding='utf-8') as fp:
            fp.write(content)
    


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


def read_text(filename):
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
    return  sentence,label

def generate_result(all_tokens, y_true, y_pred, out_path):
    with open(out_path, "w", encoding='utf-8') as fw:
        i = 0
        for sentence, y_t, y_p in zip(all_tokens, y_true, y_pred):
            if '[SEP]' in y_t or '[SEP]' in y_p:
                print(i)
                print(sentence)
                print(y_t)
                print(y_p)
            for a, b, c in zip(sentence[1:-1], y_t, y_p):
                fw.write('\t'.join([a, b, c]) + '\n')
            fw.write('\n')
            i += 1

def convert_examples_to_features_eval(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    all_tokens = []  # 存储序列，用于test自动标注还原
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids))
        all_tokens.append(ntokens)
    # with open('feature.txt','w',encoding='utf-8')as fea:
    #     for i in features:
    #         fea.write(str(i.input_ids)+'\n')
    return features, all_tokens

def clip_list(a,c):  #a为原列表，c为等分长度
    clip_back=[]
    if len(a)>c:
        for i in range(int(len(a) / c)):
            # print(i)
            clip_a = a[c * i:c * (i + 1)]
            clip_back.append(clip_a)
            # print(clip_a)
        # last 剩下的单独为一组
        last = a[int(len(a) / c) * c:]
        if last:
            clip_back.append(last)
    else:  #如果切分长度不小于原列表长度，那么直接返回原列表
        clip_back = a

    return clip_back

def get_seq(inputpath):
   f=open(inputpath,'r',encoding='utf-8')
   sentence=f.readlines()
   result=[]
   for i in sentence:
        if len(i)<509:
           result.append(i)
        else:
           result.extend(clip_list(i,509))
#    print(result)
   return result



def generate(inputpath,datapath):
    get_test(inputpath,datapath)
    get_train(inputpath,datapath)
    get_dev(inputpath,datapath)


def process_txt(raw_path,output,max_seq_length=512,eval_batch_size=30):
    filepath='wordseg_data'
    generate(raw_path,filepath)
    task_name='ner'
    dataname=filepath+'/train.txt'
    valid_txt=filepath+'/valid.txt'
    dev=filepath
    # max_seq_length=500
    bert_model='pretrain_models/siku_bert'
    output_dir='siku_aug2'
    # tempoutputpath='temp_data/'

    datalist=get_seq(raw_path)
    token=[]
    for i in datalist:
        temp=[]
        for j in i:
            temp.append(j)
        token.append(temp)
    # print(token)

    label=[]
    sentence=[]
    valid_data=readfile(filepath+'/test.txt')
    processors = {"ner":NerProcessor}
    processor = processors[task_name]()
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    eval_examples = processor.get_dev_examples(filepath)
    # eval_features = convert_examples_to_features(
    #         eval_examples, label_list,max_seq_length, tokenizer)

    #读取输出结果模型的位置
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    config = BertConfig(output_config_file)
        #加载配置文件和模型参数
    model = BertForTokenClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(output_model_file))

    #如果存在gpu,则使用gpu预测，反之使用CPU
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    n_gpu = torch.cuda.device_count()
    #将模型放在GPU上运算
    model.to(device)
    #GPU并行运算
    if n_gpu>1:
        model = torch.nn.DataParallel(model)
    #预测过程
    #加载预测内容和特征
    eval_features,alltokens = convert_examples_to_features_eval(
            eval_examples, label_list,max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    #开始预测
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            #梯度清零
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
            #通过逻辑回归取得标签
            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            #将运算结果加入列表
            for i,mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if j != max_seq_length-1:
                            if label_map[label_ids[i][j]] != "X":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            pass
                    else:
                        temp_1.pop()
                        temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
    generate_result(alltokens, y_true, y_pred, os.path.join(output, "labeled_results.txt"))

def get_output(content):
    output_text=content.replace('B','').replace('I','').replace('E','/').replace('S','/')
    return output_text

def fenci(outputpath):
    test=[]
    list3=[]
    predict=[]
    tokens=[]
    f1=open('outputdata/labeled_results.txt','r',encoding='utf-8')
    f2=open('wordseg_data/test.txt','r',encoding='utf-8')
    list1=f1.readlines()
    for i in list1:
        i=re.sub('\t','',i)
        list3.append(i)
    for k in list3:
        if k!='\n':
            tokens.append(k[-2])
        else:
            tokens.append(k)

    list2=f2.readlines()
    for j in list2:
        j=re.sub(' ','',j)
        test.append(j[0])
    # assert len(tokens)==len(test)
    tokens_txt=''.join(tokens).split('\n')
    test_txt=''.join(test).split('\n')
    combine=list(zip(test_txt,tokens_txt))
    combines=[]
    for i in range(0,len(combine)):
        temp=[]
        for j in range(0,len(combine[i][0])):
            try:
                str=combine[i][0][j]+combine[i][1][j]
                temp.append(str)
                str2=''.join(temp)
            except:
                with open(outputpath+'/'+'转化的失败句子.txt','a+',encoding='utf-8')as fp:
                    fp.write(combine[i][0]+'\n')
        combines.append(str2)
    with open(outputpath+'/result.txt','w',encoding='utf-8')as fp:
        for i in combines:
            lines=get_output(i)
            fp.write(lines+'\n')
    # print(combines)


def TCfenci(raw_path,resultpath,max_seq_length,eval_batch_size):
    process_txt(raw_path,output='outputdata',max_seq_length=max_seq_length,eval_batch_size=eval_batch_size)
    fenci(resultpath)


    
if __name__ == "__main__":
    # allprocess()
    # allprocess()
    # process_txt('wordseg_data',output='outputdata',max_seq_length=500)
    # get_seq('data')
    TCfenci(raw_path='data/清洗结果.txt',resultpath='final_result',max_seq_length=512,eval_batch_size=30)