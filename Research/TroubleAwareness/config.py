#!/usr/bin/env python
# coding=utf-8

u"コンフィグ"

__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__ = 'GPL'

import sys
import codecs


CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_KEY = ""
ACCESS_SECRET = ""

MASTER_SCREEN_NAME=""

DEBUG=True



class UserClass(object):
    u"""
    データベース操作で使うユーザーの情報を格納するクラス
    """
    def __init__(self,classify="",is_dropout=False,is_profile=False,is_read_text=False,db="db",tweets_coll="",
                 frequency_coll="",wakachi_coll="",readfiles=[""],is_discontent=False,tahara_coll="",
                 profile_coll="",tahara_coll_clean=""):
        u"""ユーザー情報を初期化する
        Args:
            classify:識別子
            is_dropout:中退した学生か
            is_profile:プロフィールは存在するか
            is_read_text:外部から読み込むファイルは必要か
            db:データベース名
            tweets_coll:ツイートを格納したコレクション名
            frequency_coll:頻度を格納したコレクション名
            wakachi_coll:分かち書きを格納したコレクション名
            readfiles:読み込むファイル名
            is_discontent:不満足なツイートをしているか
            tahara_coll:特徴語を格納したコレクション名
            profile_coll:プロフィールを格納したコレクション名
            tahara_coll_clean:整形後の特徴語を格納したコレクション名

        """
        self.classify=classify
        self.is_dropout=is_dropout
        self.is_discontent=is_discontent
        self.is_profile=is_profile
        self.is_read_text=is_read_text
        self.db=db
        self.tweets_coll=tweets_coll
        self.frecuency_coll=frequency_coll
        self.wakachi_coll=wakachi_coll
        self.readfiles=readfiles
        self.bow_coll="discontent_word"
        self.tahara_coll=tahara_coll
        self.profile_coll=profile_coll
        self.tahara_coll_clean=tahara_coll_clean


m_discontent=UserClass(classify="discontent",tweets_coll="tweets_of_discontent",wakachi_coll="wakachi_of_discontent",
                       frequency_coll="frequency_of_discontent",is_discontent=True)

m_content=UserClass(classify="content",tweets_coll="tweets_of_content",wakachi_coll="wakachi_of_content",
                       frequency_coll="frequency_of_content")

m_discontent_test=UserClass(classify="discontent_test",tweets_coll="tweets_of_discontent_test",wakachi_coll="wakachi_of_discontent_test",
                       frequency_coll="frequency_of_discontent",is_discontent=True)

m_content_test=UserClass(classify="content_test",tweets_coll="tweets_of_content_test",wakachi_coll="wakachi_of_content_test",
                       frequency_coll="frequency_of_content_test")

m_dropout=UserClass(classify="dropout",tweets_coll="tweets_of_dropout",wakachi_coll="wakachi_of_dropout",
                       frequency_coll="frequency_of_dropout",tahara_coll="tahara_of_dropout",profile_coll="profile_of_dropout",
                    tahara_coll_clean="tehara_clean_of_dropout")

m_normal=UserClass(classify="normal",tweets_coll="tweets_of_normal",wakachi_coll="wakachi_of_normal",
                       frequency_coll="frequency_of_normal",tahara_coll="tahara_of_normal",profile_coll="profile_of_normal",
                   tahara_coll_clean="tehara_clean_of_normal")

m_dropout_test=UserClass(classify="dropout_test",tweets_coll="tweets_of_dropout",wakachi_coll="wakachi_of_dropout_test",
                       frequency_coll="frequency_of_dropout_test",tahara_coll="tahara_of_dropout_test",profile_coll="profile_of_dropout_test")

m_normal_test=UserClass(classify="normal_test",tweets_coll="tweets_of_normal",wakachi_coll="wakachi_of_normal_test",
                       frequency_coll="frequency_of_normal_test",tahara_coll="tahara_of_normal_test",profile_coll="profile_of_normal_test",tahara_coll_clean="tehara_clean_of_normal_test")


class MecabDictionary():
    u"""
    Mecab辞書設定用クラス
    """
    def __init__(self,pos=0,base=0,option=""):
        u"""クラスを初期化する
        Args:
            pos:表層系の位置
            base:原型の位置
            option:Mecab使用時のオプション

        """
        self.pos=pos
        self.base=base
        self.option=option


__m_ipadic_neologd=MecabDictionary(pos=0,base=6,option="-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

__m_unidic_neologd=MecabDictionary(pos=0,base=10,option="-d /usr/local/lib/mecab/dic/mecab-unidic-neologd")

m_mecab_dic=__m_ipadic_neologd


def set_japanese_code():
    u"""標準入出力を日本語に設定する
    """
    reload(sys)
    sys.setdefaultencoding('utf-8')
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
