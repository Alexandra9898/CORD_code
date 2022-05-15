import json
import re
from collections import defaultdict

from gensim import corpora
from nltk.stem import WordNetLemmatizer


class DataLoader(object):

    def __init__(self, model, ):
        print('Init...')
        self.model = model
        self.tmp_root = './tmp_file/'
        self.stopwords_path = "./Dataset/stop_words/English_stopwords.txt"
        self.stopwords = self.stop_words()
        self.model_root = self.tmp_root + "{}_model/".format(self.model)
        self.dataset_root = './tmp_file/'  # ""D:/Support_COVID19_Research_all/Support_COVID19_Research_2022/Tmp/

    def clean_data(self, input):  # 去除所有乱七八糟的
        ret = re.sub("\W*", "", input)
        return ret

    def stop_words(self):  # 获取停词库
        with open(self.stopwords_path, "r", encoding="utf-8") as f:
            stop_words = set([self.clean_data(i.strip()) for i in f.readlines()])
            return stop_words

    def clean_list(self, words_list):  # 清理列表function
        lemmatizer = WordNetLemmatizer()
        new_words_list = []
        for word in words_list:
            new_word = self.clean_data(word.strip().lower())
            if new_word in self.stopwords or new_word == "" or re.sub("\d*", "", new_word) == "":
                continue
            else:
                new_words_list.append(lemmatizer.lemmatize(new_word))
        return new_words_list

    def clean_dictionary(self, content_list):  # 去除那些小于词频4的词
        frequency = defaultdict(int)
        for text in content_list:
            for token in text:
                frequency[token] += 1
        content_list = [[token for token in text if frequency[token] >= 4]
                        for text in content_list]
        return content_list

    def prepare_data(self):  # 准备data function
        with open(self.dataset_root + "cord_uid_ref.json", "r", encoding="utf-8") as f:
            ref_dic = json.load(f)
        content_list = []
        num = 0
        print("准备遍历施引文献...")
        with open(self.dataset_root + "cord_uid_info.txt", "r", encoding="utf-8") as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                pub_time = info['pub_time']
                if pub_time:
                    if int(pub_time.split("-")[0]) >= 2000:
                        cord_uid = info['cord_uid']
                        content = self.clean_list((info["abstract"] + " " + info["title"]).split(" "))
                        if content:
                            content_list.append(content)
                        if cord_uid in ref_dic.keys():  # 获取两千年以后的这些文章的参考文献
                            for ref in ref_dic[cord_uid]:
                                content = self.clean_list(ref["title"].split(" "))
                                if content:
                                    content_list.append(content)
                num += 1
                if num > 500:
                    break
        print("共遍历了{}篇文章。".format(num))
        print("content_list里面一共有{}行".format(len(content_list)))
        print("get dictionary and corpus...")

        if self.model in {"lda", "tfidf"}:
            content_list = self.clean_dictionary(content_list)
            dictionary = corpora.Dictionary(content_list)  # {'abdulaziz':1, 'arabia':2}
            print("去除词频<4的词之后，字典的长度是：", len(dictionary))
            corpus = [dictionary.doc2bow(text) for text in
                      content_list]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)
            dictionary.save(self.model_root + 'dictionary.dic')  # 保存生成的词典
            print("已存储字典...")  # stat dictionary.dic
            corpora.MmCorpus.serialize(self.model_root + 'corpus.mm', corpus)  # 存储语料库
            print("已存储语料库...")

        if self.model == "w2v":
            with open(self.model_root + "w2v_corpus.txt", "w", encoding="utf-8") as fw:
                for content in content_list:
                    fw.write(" ".join(content) + "\n")  # 存储语料，一行一句，用空格分隔
            print("成功存储语料库...")
