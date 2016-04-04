#!/usr/bin/env python
# coding=utf-8

u"メインの処理を行う"


import config
from data_factory import get_labeled_data
import tweet_parser as tp
from machine_learning import random_forest
from twipro import culate_9syu,kyusyu_to_mongo


__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__ = 'GPL'


def make_trouble_awareness_model():
    u"""ランダムフォレストのテスト

        Args:
            train_x:訓練データ特徴
            train_y:訓練データラベル
            test_x:テストデータ特徴
            test_y:テストデータラベル

    """
    users=[config.m_dropout,config.m_normal,config.m_dropout_test,config.m_normal_test]

    for user in users:
        to=tp.TwitterOperator(user)
        to.save_wakachi()
        to.save_tahara_format()

    pXtrain,pytrain=get_labeled_data(config.m_dropout,True)
    nXtrain,nytrain=get_labeled_data(config.m_normal,False)

    pXtest,pytest=get_labeled_data(config.m_dropout_test,True)
    nXtest,nytest=get_labeled_data(config.m_normal_test,False)

    x_train=pXtrain+nXtrain
    y_train=pytest+nytrain
    x_test=pXtest+nXtest
    y_test=pytrain+pytest

    random_forest(x_train,y_train,x_test,y_test)


def search_troubled_students():
    u"""ランダムフォレストのテスト

        Args:
            train_x:訓練データ特徴
            train_y:訓練データラベル
            test_x:テストデータ特徴
            test_y:テストデータラベル

    """
    culate_9syu()
    kyusyu_to_mongo()


def main():
    u"""ランダムフォレストのテスト

        Args:
            train_x:訓練データ特徴
            train_y:訓練データラベル
            test_x:テストデータ特徴
            test_y:テストデータラベル

    """
    config.set_japanese_code()



if __name__=='__main__':
    main()

