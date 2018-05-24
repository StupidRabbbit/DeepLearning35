#-*-coding:utf-8 -*-
import sys
import os
import jieba
from imp import reload
from sklearn.datasets.base import Bunch
import nltk
import pickle

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
#fenci完毕
# fenci()

def readfile(filepath):
        fp=open(filepath,'r',encoding='utf-8',errors='ignore')

        content=fp.read()
        fp.close()

        return content
# filepath='D:\some_data\文本分类语料库\军事249\81.txt'
# content=readfile(filepath)
# seg=jieba.cut(content,cut_all=False)
# print(' '.join(seg))
def dump_bunch():
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    wordbag_path = 'D:\some_data\\train_set\\train_set.dat'
    seg_path = 'D:\some_data\语料分词'
    catelist = os.listdir(seg_path)
    # extend是直接把内容加入list中，append是加入list对象
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        class_path = seg_path + os.path.sep + mydir + os.path.sep
        file_list = os.listdir(class_path)
        #把每一类中的文档分词都添加进去，每一篇文档都添加类别和分词
        for filepath in file_list:
            fullname = class_path + filepath
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname).strip())
    file_obj = open(wordbag_path, 'wb')

    # 对象持久化
    pickle.dump(bunch, file_obj)
    file_obj.close()
    print('构建文本对象结束')
# dump_bunch()
stopword_path='D:\some_data\\train_set\\stop_words.txt'
stpwrdlst=readfile(stopword_path).splitlines()

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def readbunchobj(path):
    file_obj=open(path,'rb')
    bunch=pickle.load(file_obj)
    file_obj.close()
    return bunch
def Writebunchobj(path,bunchobj):
    file_obj=open(path,'wb')
    pickle.dump(bunchobj,file_obj)
    file_obj.close()
path='D:\some_data\\train_set\\train_set.dat'
bunch=readbunchobj(path)
tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
transformer=TfidfTransformer()
print('shape of contents:',len(bunch.contents))
#2816篇文档

tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
#(2816, 11490)文档数，词总数
#所以这里其实是为每个文档都生成了一个向量，向量一共中有11490个词，产生了一个非常稀疏的向量
#非零的地方为文档中词的tfidf
print(tfidfspace.tdm.toarray().shape)
print(tfidfspace.tdm.toarray())
#词汇表，每一个词都有自己的编号
tfidfspace.vocabulary=vectorizer.vocabulary_
print(len(tfidfspace.vocabulary))
space_path='D:\some_data\\train_set\\tfidfspace.dat'
Writebunchobj(space_path,tfidfspace)

#多项式朴素贝叶斯计算





