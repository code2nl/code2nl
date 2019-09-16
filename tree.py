# -*- coding: utf-8 -*-
import six
import random as rnd
import token
from tokenize import tokenize
from io import BytesIO

SKP_WORD = '<sk>'
RIG_WORD = '<]>'
LFT_WORD = '<[>'

class SCode(object):
    def __init__(self, init):
        self.token_list = None
        self.type_list = None 
        self.sent_list = None

        if init is not None:
            if isinstance(init, list):
                self.set_by_list(init, None)
            elif isinstance(init, tuple):
                self.set_by_list(init[0], init[1], init[2])
            elif isinstance(init, six.string_types):
                self.set_by_str(init)
            else:
                raise NotImplementedError

    def set_by_str(self, f):
        tk_list = list(
            tokenize(BytesIO(f.strip().encode('utf-8')).readline))[1:-1]
        self.token_list = [tk.string for tk in tk_list]
        self.type_list = [token.tok_name[tk.type] for tk in tk_list]

    # well-tokenized token list
    def set_by_list(self, token_list, type_list, sent_list=None):
        self.token_list = list(token_list)
        if type_list is not None:
            self.type_list = list(type_list)
        if sent_list is not None:
            self.sent_list = list(sent_list)

    def to_list(self):
        return self.token_list

    def __str__(self):
        return ' '.join(self.to_list())

    def layout(self, add_skip=False):
        assert len(self.token_list) == len(self.type_list)
        name_list = []
        string_list = []
        number_list = []
        for tk, tp in zip(self.token_list, self.type_list):
            if tp in ('NAME',):
                name_list.append(tk)
            elif tp in ('NUMBER',):
                number_list.append(tk)
            elif tp in ('STRING',):
                string_list.append(tk)
        lay_list = [] + self.sent_list
        for each in name_list:
            id = self.get_match(each, self.sent_list)
            if id >= 0:
                lay_list[id] = 'NAME'
        for each in number_list:
            id = self.get_match(each, self.sent_list)
            if id >= 0:
                lay_list[id] = 'NUMBER'
        for each in string_list:
            id = self.get_match(each, self.sent_list)
            if id >= 0:
                lay_list[id] = 'STRING'
        return lay_list

    def get_match(self, string, sent_list): 
        for i in range(len(sent_list)):
            if string == sent_list[i]:
                return i
        return -1                



    def target(self,):
        return self.sent_list


    def norm(self, not_layout=False):
        return self


def is_code_eq(t1, t2, not_layout=False):
    if isinstance(t1, SCode):
        t1 = str(t1)
    else:
        t1 = ' '.join(t1)
    if isinstance(t2, SCode):
        t2 = str(t2)
    else:
        t2 = ' '.join(t2)
    t1 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in t1.split(' ')]
    t2 = ['\"' if it in (RIG_WORD, LFT_WORD) else it for it in t2.split(' ')]
    if len(t1) == len(t2):
        for tk1,tk2 in zip(t1,t2):
            # if not (tk1 == tk2 or tk1 == '<unk>' or tk2 == '<unk>'):
            if tk1 != tk2:
                return False
        return True
    else:
        return False
    return t1==t2


if __name__ == '__main__':
    for s in ("if base64d [ : 1 ] == b' _STR:0_ ' :".split(), "if base64d [ : 1 ] == b' _STR:0_ ' :".split(), "compressed = zlib . compress ( data )".split(), "compressed = zlib.compress(data)".split(),):
        t = SCode(s)
        print(1, t)
        print(2, t.to_list())
        print(3, ' '.join(t.layout(add_skip=False)))
        print('\n')
