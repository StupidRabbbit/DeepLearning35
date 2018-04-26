#-*-coding:utf-8 -*-
import sys
import os
import jieba
from imp import reload
reload(sys)
# sys.setdefaultencoding('utf-8')
def fenci():
    def savefile(filepath,content):
        fp=open(filepath,'w')
        fp.write(content)
        fp.close()
    def readfile(filepath):
        fp=open(filepath,'r',errors='ignore')
        content=fp.read()
        fp.close()
        return content
    corpus_path='D:\some_data\文本分类语料库'
    seg_path='D:\some_data\语料分词'
    catelist=os.listdir(corpus_path)
    # print(catelist)
    for mydir in catelist:
        class_dir=corpus_path+os.path.sep+mydir+os.path.sep
        seg_dir=seg_path+os.path.sep+mydir+os.path.sep
        print(class_dir)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        filelist=os.listdir(class_dir)

        for file_path in filelist:
            fullname=class_dir+file_path
            fullsegpath=seg_dir+file_path

            content=readfile(fullname).strip()
            content=content.replace('\r\n','').strip()
            content_seg=jieba.cut(content)
            savefile(fullsegpath,' '.join(content_seg))

            # print(fullname)

    print('语料分词完毕')
fenci()

def readfile(filepath):
        fp=open(filepath,'r',encoding='utf-8',errors='ignore')

        content=fp.read()
        fp.close()

        return content
# filepath='D:\some_data\文本分类语料库\军事249\81.txt'
# content=readfile(filepath)
# seg=jieba.cut(content,cut_all=False)
# print(' '.join(seg))





