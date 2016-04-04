#!/usr/bin/env python
# coding=utf-8

u"ツイプロに関連するモジュール"

import urllib2
import json
import re
import time
from tweet_parser import update_tweet_from_names

def get_school_names():
    u"""学校名を取得する

        Returns:
            指定した地域の学校名を返す
    """
    all_schools=set()
    pref_cd4={"鳥取":31,"島根":32,"岡山":33,"広島":34,"山口":35,"徳島":36,"香川":37,"愛媛":38,"高知":39}
    pref_cd9={"福岡":40,"佐賀":41,"長崎":42,"熊本":43,"大分":44,"宮崎":45,"鹿児島":46,"沖縄":47}

    pref_cd=pref_cd9
    for k,v in pref_cd.items():
        print v
        response=urllib2.urlopen(("http://webservice.recruit.co.jp/shingaku/school/v2?key=08a7b0676b1c6297&pref_cd="+str(v)+"&format=json&count=100"))
        results=json.load(response)
        schools=results[u"results"][u"school"]
        for school in schools:
            school_name=school[u'name'].split("｜")[0]
            school_name=re.sub(u"^(専門学校)","",school_name)
            school_name=school_name.strip()
            school_name=school_name.split(u"　")[0]
            all_schools.add(school_name)
    print len(all_schools)

    return all_schools


def get_twipro(query):
    u"""ツイプロ検索結果を返す
        Args:
            query:検索クエリ
        Returns:
            検索結果を返す
    """
    time.sleep(6)
    try:
        response=urllib2.urlopen("http://api.twpro.jp/1/search?%s"%query)
        results=json.load(response)
    except:
        print "400?"
        results=""

    return results


def get_twipro_from_school(schools,ret=""):
    u"""ツイプロから学校名を含むユーザーを取得する
        Args:
            schools:学校名
            ret:何を取得するか
        Returns:
            ユーザー名
     """
    names=set()
    for school in schools:
        request="q="+str(school)+"&num=300"
        print request
        twipro=get_twipro("q="+str(school)+"&num=300")
        if ret=="screen_name":
            try:
                users=twipro["users"]
                for user in users:
                    names.add(user["screen_name"])
            except:
                print "not users"

    return names


def culate_9syu():
    u"""九州の学生を収集する

     """
    import pickle

    schools=get_school_names()
    screen_names=get_twipro_from_school(schools,ret="screen_name")

    f=open("screen_names9","w")
    pickle.dump(screen_names,f)
    f.close()


def kyusyu_to_mongo():
    u"""九州の学生のツイ−トをデータベースに入れる
     """
    import pickle

    f=open("unique_names","r")
    unique_names=pickle.load(f)
    f.close()
    update_tweet_from_names("tweets_of_kyusyu",unique_names)
