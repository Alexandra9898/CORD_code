import argparse
import datetime
import json
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from train_model.data_loader import DataLoader  # load data
from train_model.model_training import ModelTraining  # train model to tmp file
from calculate_similarity.calculate_similarity import get_similarity  # calculate the similarity

if __name__ == '__main__':
    start_time = datetime.datetime.now()  # 开始时间
    parser = argparse.ArgumentParser()  # 创建一个解析对象description='Process some description.'
    parser.add_argument('--phase', default='test', help='the function name.')  # 为解析对象添加参数

    args = parser.parse_args()  # 用于解析
    analysis_function = args.phase.strip()  # 获取分析方法

    if analysis_function in ("lda", "w2v", "tfidf"):
        config_path = './config/{}_config.json'.format(analysis_function)
        with open(config_path, "r", encoding="utf-8") as fr:
            config = json.load(fr)
        DataLoader = DataLoader(model=analysis_function)
        DataLoader.prepare_data()  # 准备好content_list，输入到modeltraining里面训练model
        ModelTraining = ModelTraining(model=analysis_function)
        ModelTraining.get_model(**config)  # 输入content_list存储对应的model到function里面
        calculate_similarity = get_similarity(model=analysis_function)
        calculate_similarity.get_similarity(**config)  # 加载数据并获得相似度
    else:
        raise RuntimeError("There is no {} analysis function.".format(analysis_function))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done main!')
