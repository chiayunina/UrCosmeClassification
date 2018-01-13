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

# == Main Chinese segmenter function ==
def segmenter(text):
    seg_text = jieba.lcut(text)
    # Simply remove all non-Chinese letters
    for i, t in enumerate(seg_text):
        seg_text[i] = ZH.sub(r'', t)
    # Remove none value and return    
    return list(filter(None, seg_text))

def del_stops(segs):
    # Delete stopwords
    #
    # Parameters:
    # - segs: a list of tokens
    with open('stopwords.txt', 'r', encoding='utf8') as f:
        stopwords = f.readlines()
    stopwords = [w.rstrip('\n') for w in stopwords]
    
    return [w for w in segs if w not in stopwords]
    