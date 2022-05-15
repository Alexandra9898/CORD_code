from gensim import corpora, models
from gensim.models import Word2Vec
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import LineSentence


class ModelTraining(object):
    def __init__(self, model):
        self.model = model
        self.tmp_root = './tmp_file/'
        self.model_root = self.tmp_root + "{}_model/".format(self.model)

    def get_tfidf_model(self):  # 正式制作模型，并保存主题
        corpus = corpora.MmCorpus(self.model_root + "corpus.mm")
        TFIDF = TfidfModel(corpus, normalize=False)
        TFIDF.save(self.model_root + 'TFIDF_model.tfidf')
        print("成功存储模型...")
        pass

    def get_lda_model(self, **kwargs):
        show_numbers = 20
        corpus = corpora.MmCorpus(self.model_root + "corpus.mm")
        dictionary = corpora.Dictionary.load(self.model_root + "dictionary.dic")
        LDA = models.LdaModel(corpus=corpus, id2word=dictionary, random_state=1,
                              num_topics=kwargs["n_components"])  # 训练模型
        topics_list = LDA.print_topics(kwargs["n_components"], show_numbers)
        LDA.save(self.model_root + 'LDA_model.model')
        print("成功存储模型...")
        with open(self.model_root + "topics.txt", "w", encoding="utf-8") as fw:
            for topic in topics_list:
                fw.write(str(topic))

    def get_w2v_model(self, **kwargs):
        w2v_corpus_path = self.model_root + "w2v_corpus.txt"
        sentences = LineSentence(w2v_corpus_path)
        w2v_model = Word2Vec(sentences, vector_size=kwargs["n_components"], window=kwargs["window"],
                             min_count=kwargs["min_count"], workers=kwargs["workers"])
        w2v_model.save(self.model_root + 'w2v_model.model')  # 存储不同维度的模型

    def get_model(self, **config):
        if self.model == "tfidf":
            self.get_tfidf_model()
        if self.model == "lda":
            self.get_lda_model(**config)
        if self.model == "w2v":
            self.get_w2v_model(**config)
