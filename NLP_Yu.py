# -*- coding: utf-8 -*-

from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH

# 设置路径，否则会从配置文件中寻找
HANLP_DATA_PATH = "/home/fonttian/Data/CNLP"

"""
获取测试数据路径，位于$root/data/textClassification/sogou-mini，
根目录由配置文件指定,或者等于我们前面手动设置的HANLP_DATA_PATH。
"""
DATA_FILES_PATH = "E:\\data\\"


def test_data_path():
    data_path = os.path.join(HANLP_DATA_PATH, DATA_FILES_PATH)
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path


def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        return dest_path
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path


NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版',
                                'http://hanlp.linrunsoft.com/release/corpus/sogou-text-classification-corpus-mini.zip')


def train_or_load_classifier(path):
    model_path = path + '.ser'
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    classifier = NaiveBayesClassifier()
    classifier.train(sogou_corpus_path)
    model = classifier.getModel()
    IOUtil.saveObjectTo(model, model_path)
    return NaiveBayesClassifier(model)


def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))


if __name__ == '__main__':
    classifier = train_or_load_classifier(sogou_corpus_path)
    predict(classifier, "C罗压梅西内马尔蝉联金球奖 2017=C罗年")
    predict(classifier, "英国造航母耗时8年仍未服役 被中国速度远远甩在身后")
    predict(classifier, "研究生考录模式亟待进一步专业化")
    predict(classifier, "如果真想用食物解压,建议可以食用燕麦")
    predict(classifier, "通用及其部分竞争对手目前正在考虑解决库存问题")

    print("\n 我们这里再用训练好的模型连测试一下新的随便从网上找来的几个新闻标题 \n")
    predict(classifier, "今年考研压力进一步增大，或许考研正在变成第二次高考")
    predict(classifier, "张继科被刘国梁连珠炮喊醒:醒醒！奥运会开始了。")
    predict(classifier, "福特终于开窍了！新车1.5T怼出184马力，不足11万，思域自愧不如")

    predict(classifier,"博士导师证实张首晟去世 称是物理领域毁灭性损失")
    predict(classifier,"为友好合作开辟更广阔的空间")