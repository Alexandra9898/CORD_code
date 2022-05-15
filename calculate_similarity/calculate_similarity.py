import sys

sys.path.append('../')
from train_model.data_loader import DataLoader
from gensim.models import ldamodel
import json
import numpy as np
from gensim import corpora, models, similarities
from gensim.models import Word2Vec


class get_similarity(object):
    def __init__(self, model):
        print("init...")
        self.model = model
        self.DataLoader = DataLoader(self.model)
        self.tmp_root = './tmp_file/'
        self.model_root = self.tmp_root + "{}_model/".format(self.model)
        self.results_root = './Results/'
        self.dataset_root = './tmp_file/'

    def load_model(self):
        if self.model == "tfidf":
            model = models.TfidfModel.load(self.model_root + "TFIDF_model.tfidf")
        if self.model == "lda":
            model = ldamodel.LdaModel.load(self.model_root + 'LDA_model.model')
        if self.model == "w2v":
            model = Word2Vec.load(self.model_root + 'w2v_model.model')
        return model

    def get_ref_dic(self):
        with open(self.dataset_root + "cord_uid_ref.json", "r", encoding="utf-8") as f:
            ref_dic = json.load(f)
        return ref_dic

    def get_dictionary(self):
        dictionary = corpora.Dictionary.load(self.model_root + "dictionary.dic")
        return dictionary

    def get_index2vec(self):
        index2vec = set(self.load_model().wv.index_to_key)
        return index2vec

    def get_vector(self, word_list, w2v_model):
        vector_list = []
        for word in word_list:
            if word in self.get_index2vec():
                vector_list.append(w2v_model.wv[word])
        if vector_list:
            vector_sentence = np.mean(np.array(vector_list), axis=0)
            return vector_sentence
        else:
            return []

    def get_tfidf_median_simi(self, dictionary, texts_ref, texts_info, TFIDF, dic_length):
        bow_ref = [i for i in [dictionary.doc2bow(text) for text in texts_ref] if i != []]
        if bow_ref:
            tfidf_vector_ref = TFIDF[bow_ref]
            index = similarities.MatrixSimilarity(tfidf_vector_ref, num_features=dic_length)
            bow_info = dictionary.doc2bow(texts_info)
            if bow_info:
                tfidf_vector_info = TFIDF[bow_info]  # 获取文章的向量
                sims = list(index[tfidf_vector_info])
                median_simi = np.median(sims).item()
                return median_simi

    def get_lda_median_simi(self, dictionary, texts_ref, texts_info, LDA, n_components):
        bow_ref = [i for i in [dictionary.doc2bow(text) for text in texts_ref] if i != []]  # 用字典把它变成词袋
        if bow_ref:
            LDA_vector_ref = LDA[bow_ref]
            index = similarities.MatrixSimilarity(LDA_vector_ref, num_features=n_components)
            bow_info = dictionary.doc2bow(texts_info)
            if bow_info:
                LDA_vector_info = LDA[bow_info]  # 获取文章的向量
                sims = list(index[LDA_vector_info])
                median_simi = np.median(sims).item()
                return median_simi

    def get_w2v_median_simi(self, texts_ref, texts_info, w2v_model):
        w2v_vector_info = self.get_vector(texts_info, w2v_model)  # 返回的可能是None或者是一个300维度向量
        if w2v_vector_info != []:  # 如果是一个300维的向量
            w2v_vector_refs = []
            for text_ref in texts_ref:
                w2v_vector_ref = self.get_vector(text_ref, w2v_model)
                if w2v_vector_ref != []:
                    w2v_vector_refs.append(w2v_vector_ref)
            if w2v_vector_refs:
                simi_all = np.dot(np.array(w2v_vector_refs), np.array(w2v_vector_info)) / np.linalg.norm(
                    np.array(w2v_vector_info)) / np.linalg.norm(np.array(w2v_vector_refs), axis=1)
                median_simi = np.median(simi_all).item()
                return median_simi

    def get_similarity(self, **kwargs):
        ref_dic = self.get_ref_dic()
        if self.model == "tfidf":
            TFIDF = self.load_model()
            dictionary = self.get_dictionary()
            dic_length = len(dictionary)
        if self.model == "lda":
            LDA = self.load_model()
            dictionary = self.get_dictionary()
            n_components = kwargs["n_components"]
        if self.model == "w2v":
            w2v_model = self.load_model()
        print("已加载模型，即将获取向量...")
        sims_dict = {}
        with open(self.dataset_root + "cord_uid_info.txt", "r", encoding="utf-8") as fp:
            num = 0
            while True:
                num += 1
                if num > 500:
                    break
                line = fp.readline().strip()
                if not line:
                    break
                info = json.loads(line)
                pub_time = info["pub_time"]
                if pub_time:
                    if int(pub_time.split("-")[0]) >= 2000:  # 这篇文章的时间在2000年周后
                        cord_uid = info["cord_uid"]
                        if cord_uid in ref_dic:  # 如果这篇文章有参考文献
                            content_info = info["title"] + " " + info["abstract"]
                            texts_info = self.DataLoader.clean_list(content_info.split(" "))
                            if texts_info:  # 如果文章里面有东西
                                refs = ref_dic[cord_uid]
                                texts_ref = []
                                for ref in refs:
                                    text_ref = self.DataLoader.clean_list(ref['title'].split(" "))
                                    if text_ref:
                                        texts_ref.append(text_ref)
                                if texts_ref:  # 如果texts不是空的列表
                                    if self.model == "tfidf":
                                        median_simi = self.get_tfidf_median_simi(dictionary, texts_ref, texts_info,
                                                                                 TFIDF, dic_length)
                                    if self.model == "lda":
                                        median_simi = self.get_lda_median_simi(dictionary, texts_ref, texts_info, LDA,
                                                                               n_components)
                                    if self.model == "w2v":
                                        median_simi = self.get_w2v_median_simi(texts_ref, texts_info, w2v_model)
                                    if median_simi != None:
                                        sims_dict[cord_uid] = median_simi
        print("共生成{}个相似度值".format(len(sims_dict)))
        with open(self.results_root + "cord_uid_{}_similarity.json".format(self.model), "w", encoding="utf-8") as fw:
            json.dump(sims_dict, fw, ensure_ascii=False)
