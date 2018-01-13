# Chinese Segmenter
import re
import jieba

ZH = re.compile(u'[^\u4E00-\u9FA5]')

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

@run_once    
def set_dict():
    jieba.set_dictionary('dict.txt.big.txt')    

set_dict()    

def segmenter(text):
    seg_text = jieba.lcut(text)
    # Simply remove all non-Chinese letters
    for i, t in enumerate(seg_text):
        seg_text[i] = ZH.sub(r'', t)
    # Remove none value and return    
    return list(filter(None, seg_text))
