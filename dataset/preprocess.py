import html
import re
import multiprocessing as mp
from multiprocessing import Pool
import sys
from  config import hparams as hp
in_path = '../data/搜狗文本分类语料库已分词.txt'
out_path = '../data/corpus_pre.txt'

def preprocess(line):
    lines = []
    sent = line
    split = sent.split(hp.split_mark)
    is_pure_english = False
    for s in split:
        if s =='\n':
            continue
        is_pure_english = judge_pure_english(s.strip())
        if is_pure_english:
            break
    if is_pure_english is False:
        if 5 < len(split) and len(split) < hp.enc_maxlen - 2:  # 只保留长度刚好的句子 其它舍弃掉
            sent = html.unescape(sent.strip())  # 将字符串中的html实体转换成html标签
            lines.append(sent + '\n')
    return 1, lines




def has_hanzi(line):
  zhmodel = re.compile(u'[\u4e00-\u9fa5]')
  match = zhmodel.search(line)
  if match:
    return True
  else:
    return False


def judge_pure_english(keyword):
    return all(ord(c) < 128 for c in keyword)

def html_unescape(_str):  # 将字符串中的html实体转换成html标签
    return html.unescape(_str)

def progbar(i, n, size=55):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar

def stream(message):
    sys.stdout.write(f"\r{message}")


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_path", required=True, type=str)
    # parser.add_argument("-o", "--output_path", required=True, type=str)
    # args = parser.parse_args()

    import os

    print(os.getcwd()) # 获取当前工作目录路径
    p = Pool(mp.cpu_count() - 1)
    print(mp.cpu_count() - 1)

    results = []

    data = open(in_path, 'r', encoding='utf-8').readlines()
    out_lines = []
    count = 0
    for line in data:
        #preprocess(line)#测试用
        # 一个进程执行一句
        res = p.apply_async(preprocess, (line,))
        results.append(res)

    p.close()
    p.join()

    for i, res in enumerate(results):

        line_count, lines = res.get()  # linecount表示一个线程执行一句， 这一句分出的数量； lines表示这一句最后的输出
        out_lines += lines
        count += line_count

        bar = progbar(i, len(data))
        message = f'{bar} {i}/{len(data)} '
        stream(message)

    sort_lines = sorted(out_lines, key=lambda i: len(i), reverse=True)
    sort_lines = set(sort_lines)
    print("原始总共：{}句，分句后{}，删除{}句，剩余{}句！".format(len(data), count, count-len(sort_lines), len(sort_lines)))
    # 预处理后的句子写入到新的文件中去
    with open(out_path, 'w',encoding='utf-8') as f:
        f.writelines(sort_lines)

