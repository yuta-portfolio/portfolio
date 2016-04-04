#!/usr/bin/env python
# coding=utf-8

u"データを作成・加工するモジュール"

import datetime
import random
from time import time
import pandas as pd
import skflow
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import config
from config import set_japanese_code
from function_common import load_from_mongo,dprint,delete_mongo,extract_time, load_stopwords_old


__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__ = 'GPL'


def make_profile_of_dropout():
    u"""学生の中退日時を設定するメソッド

    """
    import pymongo

    delete_mongo("db","profile_of_dropout")

    #MongoDB準備
    client=pymongo.MongoClient()
    db=client["db"]
    coll=db["profile_of_dropout"]

    coll.save({'_id':u"",'date_of_dropout':"2015:09:14:22:57:33"})
    coll.save({'_id':u"",'date_of_dropout':"2015:07:28:20:29:09"})
    coll.save({'_id':u"",'date_of_dropout':"2015:09:29:00:00:00"})
    coll.save({'_id':u"",'date_of_dropout':"2015:10:02:17:15:19"})
    coll.save({'_id':u"",'date_of_dropout':"2015:10:05:11:44:10"})
    coll.save({'_id':u"",'date_of_dropout':"2015:10:09:10:31:18"})
    coll.save({'_id':u"",'date_of_dropout':"2015:10:05:17:28:40"})
    coll.save({'_id':u"",'date_of_dropout':"2015:10:13:10:11:59"})
    coll.save({'_id':u"",'date_of_dropout':"2015:09:19:12:50:34"})
    coll.save({'_id':u"",'date_of_dropout':"2015:11:24:00:00:00"})


def make_profile_of_normal():
    u"""学生のツイートを収集開始時点を設定するメソッド

    """
    import pymongo

    delete_mongo("db","profile_of_normal")

    #MongoDB準備
    client=pymongo.MongoClient()
    db=client["db"]
    coll=db["profile_of_normal"]

    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})
    coll.save({'_id':u"",'date_of_normal':"2015:10:01:00:00:00"})


def add_end_tweet_point():
    u"""ツイート収集終了時点を設定する

    """
    import pymongo

    profiles=load_from_mongo("db","profile_of_dropout")
    print "length of dropout:",len(profiles)

    count=0

    client = pymongo.MongoClient()
    db=client[config.m_dropout.db]
    coll=db[config.m_dropout.profile_coll]

    #中退者のツイート収集終了時点を設定
    for profile in profiles:
        y=extract_time(profile["date_of_dropout"],"Y")
        m=extract_time(profile["date_of_dropout"],"M")
        d=extract_time(profile["date_of_dropout"],"D")
        dropout_day=datetime.date(int(y),int(m),int(d))
        day100=datetime.timedelta(99)
        start_day=dropout_day-day100
        count_start="{0}:{1:0>2}:{2:0>2}:00:00:00".format(start_day.year,start_day.month,start_day.day)
        coll.save({"_id":profile["_id"],"date_of_dropout":profile["date_of_dropout"],"date_of_start":count_start})
        tweets=load_from_mongo(config.m_dropout.db,config.m_dropout.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':count_start,'$lte':profile["date_of_dropout"]}}]})
        if len(tweets)==0:
            tweets=load_from_mongo("db","tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':count_start,'$lte':profile["date_of_dropout"]}}]})
        count+=len(tweets)

    profiles=load_from_mongo("db","profile_of_normal")
    print "length of normal:",len(profiles)

    count=0

    client = pymongo.MongoClient()
    db=client[config.m_normal.db]
    coll=db[config.m_normal.profile_coll]

    #普通の学生のツイート収集終了時点を設定
    for profile in profiles:
        normal_days="2015:12:29:23:59:59"
        y=extract_time(normal_days,"Y")
        m=extract_time(normal_days,"M")
        d=extract_time(normal_days,"D")
        dropout_day=datetime.date(int(y),int(m),int(d))
        day100=datetime.timedelta(99)
        start_day=dropout_day-day100
        count_start="{0}:{1:0>2}:{2:0>2}:00:00:00".format(start_day.year,start_day.month,start_day.day)
        coll.save({"_id":profile["_id"],"date_of_dropout":normal_days,"date_of_start":count_start})
        tweets=load_from_mongo(config.m_normal.db,config.m_normal.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':count_start,'$lte':normal_days}}]})
        count+=len(tweets)


def print_top_words(model, feature_names, n_top_words):
    u"""ldaのトピック上位を表示する

        Args:
            model:ldaモデル
            feature_names:特徴語名
            n_top_words:何ワード表示するか
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def make_document_corpus(wakachi):
    u"""分かち書きをリスト化して返す

        Args:
            wakachi:分かち書き
        Returns:
            docs:分かち書きをリスト化したもの
    """
    docs=[]
    for doc in wakachi:
        docs.append(doc["wakachi"])
    return docs


def bow_to_lsi(user,layer=20,):
    u"""BoWを次元削減する

    参考 http://qiita.com/yasunori/items/31a23eb259482e4824e2

        Args:
            user:Userクラス
            layer:次元数
    """
    set_japanese_code()

    stopwords= load_stopwords_old()
    classify=user.classify


    wakachi=load_from_mongo(user.db,user.wakachi_coll)
    wakachi2=load_from_mongo("db","content_timeline")
    wakachi.extend(wakachi2)
    tweets=make_document_corpus(wakachi)

    vectorizer=TfidfVectorizer(min_df=4,max_df=0.80,stop_words=stopwords,ngram_range=(1,3))

    svd_model=TruncatedSVD(n_components=20,
                           algorithm="randomized",n_iter=10,random_state=42)
    svd_transformer=Pipeline([('tfidf',vectorizer),('svd',svd_model)])
    query_vector=svd_transformer.transform(tweets)


def wakachi_to_lsi(p_wakachi,n_wakachi,layer=20):
    u"""分かち書きを次元削減する

    参考 https://medium.com/@adi_enasoaie/easy-lsi-pipeline-using-scikit-learn-a073f2484408#.1h9fqqmky

        Args:
            p_wakachi:pos分かち書き
            n_wakachi:neg分かち書き
            layer:次元数
        Returns:
            query_vector:テストベクトル
            svd_transformer:tf-idf→svdのパイプライン
    """

    set_japanese_code()

    stopwords= load_stopwords_old()
    wakachi=p_wakachi+n_wakachi
    tweets=make_document_corpus(wakachi)

    #ツイートにゴミが入ってないかチェック
    dprint(tweets)

    #TF-IDF
    vectorizer=TfidfVectorizer(ngram_range=(1,3),analyzer="word",binary=False)
    #SVD
    svd_model=TruncatedSVD(n_components=layer,
                           algorithm="randomized",n_iter=10,random_state=42)
    svd_transformer=Pipeline([('tfidf',vectorizer),('svd',svd_model)])

    tweets_p=make_document_corpus(p_wakachi)

    #訓練用データ
    svd_matrix=svd_transformer.fit_transform(tweets_p)

    #テスト用データ
    query_vector=svd_transformer.transform(tweets)

    return query_vector,svd_transformer


def create_train_vector(p_user,n_user,output):
    u"""訓練用ベクトルを作成する

    posのラベルは1、negのラベルは-1

        Args:
            p_user:posユーザー
            n_user:negユーザー
            output:ベクトルを保存するファイル名
        Returns:
            svd_matrix:tf-idf→svdのパイプライン
    """
    p_wakachi=load_from_mongo(p_user.db,p_user.wakachi_coll)
    n_wakachi=load_from_mongo(n_user.db,n_user.wakachi_coll)
    wakachi=p_wakachi+n_wakachi
    matrix,svd_matrix=wakachi_to_lsi(p_wakachi,n_wakachi)

    label_t='1'
    label_f='-1'

    ids=[w["_id"] for w in wakachi]
    t=[label_t]*len(p_wakachi)
    f=[label_f]*len(n_wakachi)
    tf=t+f
    pd_matrix=pd.DataFrame([ids,tf,list(matrix[:,0]),list(matrix[:,1]),list(matrix[:,2]),list(matrix[:,3]),list(matrix[:,4]),list(matrix[:,5])
                             ,list(matrix[:,6]),list(matrix[:,7]),list(matrix[:,8]),list(matrix[:,9]),list(matrix[:,10]),list(matrix[:,11])
                             ,list(matrix[:,12]),list(matrix[:,13]),list(matrix[:,14]),list(matrix[:,15]),list(matrix[:,16]),list(matrix[:,17])
                             ,list(matrix[:,18]),list(matrix[:,19])]).T

    vs=["v{0}".format(str(x)) for x in range(0,20)]
    pd_matrix.columns=list(["id"]+["tf"]+vs)
    pd_matrix["id"]=pd_matrix.ix[:,"id"].astype(int)

    #パラメータがゼロの行は削除する
    pd_matrix = pd_matrix[pd_matrix.v0 != 0]

    #TFの要素数を揃える
    t_list=pd_matrix[pd_matrix["tf"]==label_t]
    f_list=pd_matrix[pd_matrix["tf"]==label_f]
    if(len(t_list))>(len(f_list)):
        dprint("In t > f")
        t_list=t_list.loc[random.sample(t_list.index,len(f_list))]
    elif(len(f_list))>(len(t_list)):
        dprint("In f > t")
        f_list=f_list.loc[random.sample(f_list.index,len(t_list))]
    pd_matrix=pd.concat([t_list,f_list])

    pd_matrix.to_csv(output,index=False)

    return svd_matrix


def create_test_vector(user,svd_transformer,savefile):
    u"""テストベクトルを作成する

        Args:
            user:ユーザークラス
            svd_transformer:次元削減パイプライン
            savefile:ベクトルを保存するファイル名
    """
    wakachi=load_from_mongo(user.db,user.wakachi_coll)
    tweets=make_document_corpus(wakachi)
    matrix=svd_transformer.transform(tweets)


    ids=[w["_id"] for w in wakachi]
    tf=["True"]*len(wakachi)
    pd_matrix=pd.DataFrame([ids,tf,list(matrix[:,0]),list(matrix[:,1]),list(matrix[:,2]),list(matrix[:,3]),list(matrix[:,4]),list(matrix[:,5])
                             ,list(matrix[:,6]),list(matrix[:,7]),list(matrix[:,8]),list(matrix[:,9]),list(matrix[:,10]),list(matrix[:,11])
                             ,list(matrix[:,12]),list(matrix[:,13]),list(matrix[:,14]),list(matrix[:,15]),list(matrix[:,16]),list(matrix[:,17])
                             ,list(matrix[:,18]),list(matrix[:,19])]).T
    vs=["v{0}".format(str(x)) for x in range(0,20)]
    pd_matrix.columns=list(["id"]+["tf"]+vs)
    #dprint(type(pd_matrix))
    pd_matrix["id"]=pd_matrix.ix[:,"id"].astype(int)

    pd_matrix = pd_matrix[pd_matrix.v0 != 0]
    pd_matrix.to_csv(savefile,index=False)


def merge_vector_csv(pos_data, neg_data, save_file):
    u"""特徴ベクトルを保存したCSVをマージする

        Args:
            pos_data:posファイル名
            neg_data:negファイル名
            save_file:マージしたファイル名
    """
    pd_pos=pd.read_csv(pos_data,sep=',',header=0)
    pd_neg=pd.read_csv(neg_data,sep=',',header=0)
    pd_pn=pd.concat([pd_pos,pd_neg])
    pd_pn.to_csv(save_file,index=False)


def get_labeled_data(user,label):
    u"""ラベル付きデータを返す

        Args:
            user:ユーザークラス
            label:ラベル
        Returns:
            tweets:ユーザーのツイートをリスト化
            labels:ラベル
    """
    wakachi=load_from_mongo(user.db,user.wakachi_coll)
    tweets=make_document_corpus(wakachi)
    label=label
    labels=[label]*len(wakachi)

    return tweets,labels


class DenseTransformer(TransformerMixin):
    u"""
    これをかませないとDNNがバグる
    """
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    def fit(self, X, y=None, **fit_params):
        return self


def create_ngram_model(params=None):
    u"""NaiveBayseの訓練モデルを作成
        Args:
            params:パラメータ
        Returns:
            pipleline:tfidf→naivebyseのパイプライン
    """
    tfidf_ngrams=TfidfVectorizer(ngram_range=(1,3),analyzer="word",binary=False)
    clf=MultinomialNB()
    pipeline=Pipeline([('vect',tfidf_ngrams),('clf',clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline

def create_ngram_model_rf(params=None):
    u"""RandomForestの訓練モデルを作成
        Args:
            params:パラメータ
        Returns:
            pipleline:tfidf→randomforestのパイプライン
    """
    tfidf_ngrams=TfidfVectorizer(ngram_range=(1,3),min_df=1,max_df=0.5,analyzer="word",binary=False)
    clf=RandomForestClassifier(n_estimators=500)
    pipeline=Pipeline([('vect',tfidf_ngrams),('clf',clf)])
    if params:
        pipeline.set_params(**params)

    return pipeline

def create_ngram_model_svm(params=None):
    u"""SVMの訓練モデルを作成
        Args:
            params:パラメータ
        Returns:
            pipleline:tfidf→svmのパイプライン
    """
    tfidf_ngrams=TfidfVectorizer(ngram_range=(1,1),analyzer="word",binary=False,smooth_idf=False,use_idf=False)
    clf=LinearSVC(class_weight="balanced")
    pipeline=Pipeline([('vect',tfidf_ngrams),('clf',clf)])
    if params:
        pipeline.set_params(**params)

    return pipeline


def create_ngram_model_kmean(params=None):
    u"""k平均法の訓練モデルを作成
        Args:
            params:パラメータ
        Returns:
            pipleline:tfidf→kmeanのパイプライン
    """
    tfidf_ngrams=TfidfVectorizer(ngram_range=(1,2),analyzer="word",binary=False,smooth_idf=False,use_idf=False)
    clf=KMeans(n_clusters=5,random_state=10)
    pipeline=Pipeline([('vect',tfidf_ngrams),('clf',clf)])
    if params:
        pipeline.set_params(**params)

    return pipeline


def create_ngram_model_dnn(params=None):
    u"""深層学習のモデルを作成

        Args:
            params:パラメータ
        Returns:
            pipeline:cntvect→dnnのパイプライン
    """
    tfidf_ngrams=TfidfVectorizer(ngram_range=(1,1),analyzer="word",binary=False,smooth_idf=False,use_idf=False)
    cnt=CountVectorizer()
    clf=skflow.TensorFlowDNNClassifier(hidden_units=[10,20,10],n_classes=3)
    pipeline=Pipeline([('vect',cnt),('to_dense',DenseTransformer()),('clf',clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline


def create_ngram_model_pca(params=None):
    u"""主成分分析のモデルを作成

        Args:
            params:パラメータ
        Returns:
            pipleline:tfidf→pcaのパイプライン
    """
    from sklearn import decomposition
    tfidf_ngrams=CountVectorizer()
    pca=decomposition.KernelPCA(n_components=50)
    pipeline=Pipeline([('vect',tfidf_ngrams),('pca',pca)])
    if params:
        pipeline.set_params(**params)

    return pipeline


def wakachi_lda(user):
    u"""トピック分析をする

        Args:
            user:ユーザークラス
    """
    n_top_words=50

    set_japanese_code()

    classify=user.classify

    stopwords= load_stopwords_old()

    # TfIdfVectorizerをLDA用に作る
    vectorizer=TfidfVectorizer(min_df=1,max_df=0.80,stop_words=stopwords)

    wakachi=load_from_mongo(user.db,user.wakachi_coll)
    tweets=[]

    for words in wakachi:
            tweets.append(words["wakachi"])
    X=vectorizer.fit_transform(tweets)

    num_samples,num_features=X.shape
    print ("#samples: %d, #features: %d" % (num_samples,num_features))

    # TfidfVectorizerをLDAに使う
    print "Fitting LDA models with tfidf features, n_samples=%d and n_features=%d..."%(num_samples,num_features)
    lda=LatentDirichletAllocation(n_topics=n_top_words,max_iter=5,learning_method='online',learning_offset=50,random_state=0)
    t0=time()
    lda.fit(X)

    print "done in %0.3fs."%(time()-t0)
    print "\nTopics in LDA model:"
    tf_features_name=vectorizer.get_feature_names()
    print_top_words(lda,tf_features_name,10)


def tahara_features(classify="dropout", is_classify="dropout", is_80=False):
    u"""田原アルゴリズムに基づいて特徴語を抽出し、ユーザーベクトルを作成する

        Args:
            calassify:適用するユーザーのクラス
            is_classify:どのユーザークラスらしさでベクトルを作成するか
            is_80:上位80語制限
    """

    import pymongo
    from decimal import Decimal,Context
    import math
    from function_common import get_word_point

    context=Context()
    count=0

    if classify == "dropout":
        profiles=load_from_mongo(config.m_dropout.db,config.m_dropout.profile_coll)
    elif classify=="normal":
        profiles=load_from_mongo(config.m_normal.db,config.m_normal.profile_coll)
    elif classify=="dropout_test":
        profiles=load_from_mongo(config.m_dropout_test.db,config.m_dropout_test.profile_coll)
        print len(profiles)
    elif classify=="normal_test":
        profiles=load_from_mongo(config.m_normal_test.db,config.m_normal_test.profile_coll)
        print len(profiles)
    else:
        print "unknown classify ",classify
        return

    if is_classify =="dropout":
            dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,True)
            names=[]
            for d in dropouts.sort("rate",pymongo.DESCENDING):
                count+=1
                names.append(d["_id"])
                if count >=1000:
                    break
            if(is_80==True):
                print "name,dropout,{0}".format(
                ",".join([str(n) for n in names if get_word_point(n) > 0]))
            else:
                print "name,dropout,{0}".format(",".join([str(n) for n in names]))

    for profile in profiles:
        if classify == "dropout" or classify == "normal":
            wui=load_from_mongo("db","users_of_tehara",return_cursor=False,criteria={'_id':profile['_id']})
        elif classify == "dropout_test" or classify == "normal_test":
            wui=load_from_mongo("db","users_of_tehara_test",return_cursor=False,criteria={'_id':profile['_id']})
        else:
            print "unknown classify ",classify
            return

        if (wui is None) or (len(wui)==0):
            continue
        words= wui[0][u'words']
        wui_wci=0
        w2ui=0
        w2ci=0

        if is_classify =="dropout":
            dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,True)
        else:
            dropouts=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,True)

        if classify=="dropout":
            tweets=load_from_mongo(config.m_dropout.db,config.m_dropout.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            if len(tweets)==0:
                tweets=load_from_mongo(config.m_dropout.db,"tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            tmp_sim_dropouts=[]
            tmp_sim_dropouts.append(profile["_id"])
            tmp_sim_dropouts.append(-1)
        elif classify == "normal":
            tweets=load_from_mongo(config.m_normal.db,config.m_normal.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            tmp_sim_dropouts=[]
            tmp_sim_dropouts.append(profile["_id"])
            tmp_sim_dropouts.append(1)
        elif classify == "dropout_test":
            tweets=load_from_mongo(config.m_dropout_test.db,config.m_dropout_test.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            if len(tweets)==0:
                tweets=load_from_mongo(config.m_dropout_test.db,"tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            tmp_sim_dropouts=[]
            tmp_sim_dropouts.append(profile["_id"])
            tmp_sim_dropouts.append(-1)
        elif classify == "normal_test":
            tweets=load_from_mongo(config.m_normal_test.db,config.m_normal_test.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
            tmp_sim_dropouts=[]
            tmp_sim_dropouts.append(profile["_id"])
            tmp_sim_dropouts.append(-1)

        # 実際につぶやきがあった日数をカウント
        day_counter=set()
        for tweet in tweets:
            y=extract_time(tweet["created_at"],"Y")
            m=extract_time(tweet["created_at"],"M")
            d=extract_time(tweet["created_at"],"D")
            days=y,":",m,":",d
            day_counter.add(days)
        total_days=len(day_counter)

        if total_days == 0:
            print "Error! ",profile["_id"],"has 0 tweets."
            continue

        tmp_sim_dropouts=[]
        count=0

        for d in dropouts.sort("rate",pymongo.DESCENDING):

            count+=1
            word=[w for w in words if w["name"] == d["_id"]]
            if len(word)==0:
                word=[{"count":0,"name":d["_id"],"days":0}]
            else:
                pass

            cnt=Decimal(word[0]["count"])
            rate=Decimal(d["rate"])
            days=Decimal(word[0]["days"])
            d_point=Decimal(get_word_point(d["_id"]))

            if d_point == 0:
                if is_classify=="dropout":
                    if count > 1000:
                        if is_80==True:
                            break
                        else:
                            pass
                    else:
                        if is_80==True:
                            continue
                        else:
                            pass
            td=Decimal(total_days)

            if is_classify=="dropout":
                tmp_wui_wci=Decimal(days/td)
                wui_wci+=tmp_wui_wci
                tmp_w2ui=context.power(cnt,2)
                w2ui+=tmp_w2ui
                tmp_w2ci=context.power(rate,2)
                w2ci+=tmp_w2ci

            else:
                tmp_wui_wci=Decimal(cnt*rate)
                wui_wci+=(cnt*rate)
                tmp_w2ui=Decimal(math.pow((Decimal(cnt)),2))
                w2ui+=math.pow((cnt),2)
                tmp_w2ci=Decimal(math.pow(Decimal(rate),2))
                w2ci+=math.pow(rate,2)

            if count >= 1000:
                break

            if tmp_wui_wci == 0 or tmp_w2ci == 0 or tmp_w2ui == 0 or cnt ==0:
                tmp_sim_dropouts.append(0)
            else:
                tmp_sim_dropout=tmp_wui_wci
                tmp_sim_dropouts.append(float(tmp_sim_dropout))

        print profile["_id"],",True,{0}".format(
            ",".join([str(t) for t in tmp_sim_dropouts])
        )


def make_stopwords():
    u"""コピペ用ストップワードを作成して表示

    """
    import mojimoji
    import cnvk
    stopwords=set()
    hira=u"あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもらりるれろやゐゆゑよわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっゔ"
    kata=[]
    for h in hira:
        kata.append(cnvk.convert(h,cnvk.HIRA2KATA,cnvk.Z_KATA))
    kata.append(u"ヴ")
    hankata=[]
    for k in kata:
        hankata.append(mojimoji.zen_to_han(k))
    kazu=u"0123456789"
    stopwords.add(u"10")
    stopwords.add(u"11")
    stopwords.add(u"12")
    stopwords.add(u"13")
    stopwords.add(u"14")
    stopwords.add(u"15")
    stopwords.add(u"16")
    stopwords.add(u"17")
    stopwords.add(u"18")
    stopwords.add(u"19")
    stopwords.add(u"20")
    stopwords.add(u"１０")
    stopwords.add(u"１１")
    stopwords.add(u"１２")
    stopwords.add(u"１３")
    stopwords.add(u"１４")
    stopwords.add(u"１５")
    stopwords.add(u"１６")
    stopwords.add(u"１７")
    stopwords.add(u"１８")
    stopwords.add(u"１９")
    stopwords.add(u"２０")
    zenkazu=mojimoji.han_to_zen(kazu)
    kazukan=u"一二三四五六七八九十百千万億兆"
    minialpha=u"abcdefghijklmnopqlstuvwxyz"
    bigalpha=u"ABCDEFGHIJKLMNOPQLSTUVWXYZ"
    han_minialpha=mojimoji.han_to_zen(minialpha)
    han_bigalpha=mojimoji.han_to_zen(bigalpha)
    hiramoji=[u"する",u"なる",u"てる",u"れる",u"やる",u"いる",u"さん",u"なん",u"くん",u"それ",u"こと",\
              u"ちゃん",u"ある",u"これ",u"して",u"くれる",u"くださる",u"そう",u"せる",u"した",u"いか",\
              u"ので",u"よう",u"てるん",u"もん",u"られる",u"あそこ",u"あたり",u"あちら",u"あっち",u"あと",\
              u"あな",u"あなた",u"あれ",u"いくつ",u"いつ",u"いま",u"いろいろ",u"うち",u"おおまか",u"おまえ",u"おれ",
              u"がい",u"かく",u"かたちの",u"かやの",u"から",u"がら",u"きた",u"こせ",u"ここ",u"こっち",u"こと",u"ごと",\
              u"こちら",u"これ",u"これら",u"ごろ",u"さまざま",u"さらい",u"しかた",u"しよう",u"すか",u"ずつ",u"すね",\
              u"そう",u"そこ",u"そちら",u"そっち",u"そで",u"それ",u"それぞれ",u"それなり",u"たくさん",u"たち",u"たび",\
              u"ため",u"ちゃ",u"てん",u"とおり",u"とき",u"どこ",u"どこか",u"ところ",u"どちら",u"どれ",u"なか",u"なかば",\
              u"なに",u"など",u"なん",u"はじめ",u"はず",u"はるか",u"ひと",u"ひとつ",u"ふく",u"ぶり",u"べつ",u"へん",u"べん",\
              u"ほう",u"ほか",u"まさ",u"まし",u"まとも",u"まま",u"みたい",u"みつ",u"みなさん",u"みんな",u"もと",u"もの",\
              u"もん",u"やつ",u"よう",u"よそ",u"わけ",u"わたし",u"くる",u"すぎる",u"れる",u"いう",u"くださる",u"ちゃう",\
              u"つく",u"せる",u"てるん",u"すぎ",u"ところ",u"おれ",u"ぼく",u"わたし",u"てる",u"しまう",u"みる",
              ]

    katamoji=[]
    for h in hiramoji:
        katamoji.append(cnvk.convert(h,cnvk.HIRA2KATA,cnvk.Z_KATA))
    han_katamoji=[]
    for k in katamoji:
        han_katamoji.append(mojimoji.zen_to_han(k))

    kanmoji=["笑","今","気","今日","明日","方","人","俺","私","僕","時","思う","行く","言う","見る","出す","年","月","日","分","秒","週","火","水","木","金","土","国","都",\
             "道","府","県","市","区","町","村","各","第","何","的","度","達","誰","者","類","用","別","等","際","系","品","化","所","毎","回","匹","個","席","束","歳","円","毎",\
             "前","後","左","右","次","先","春","夏","秋","冬","下記","上記","時間","今回","前回","場合","自分","ヶ所","ヵ所","カ所","箇所","ヶ月","カ月","箇月","名前","本当","確か","時点",\
             "様々","結局","半ば","以前","以後","以降","未満","以上","以下","毎日","自体","何人","手段","感じ","同じ","点","君"]

    h_kigou=cnvk.H_KIGO
    kigou=[]
    for h in h_kigou:
        for x in h:
            kigou.append(x)
    kigou.append(u"ω")
    kigou.append(u'ー')
    kigou.append(u"д")

    #参考 内容推測に適したキーワード抽出のための日本語ストップワード(https://www.jstage.jst.go.jp/article/jjske/12/4/12_511/_pdf)
    kokubu_words=[u"ない",u"高い",u"多い",u"少ない","強い","大きい","小さい","長い","ながい",
                  u"良い",u"よい",u"いい","悪い",
                  u"ある","いる","なる","行く","いく","来る","とる",
                  "見る","みる","言う","いう","得る","過ぎる","すぎる",
                  "する","やる","行なう","行う","おこなう","出来る","できる",
                  "おもう","思う","考える","かんがえる","わかる","見える",
                  "知る","しれる","いえる","示す","述べる","書く","かく","よる",
                  "異なる","違う","ちがう","くらべる",
                  "入れる","出る","でる","入る","はいる",
                  "使う","用いる","もちいる","持つ","もつ","作る","つくる",
                  "なす","起こる","おこる","つく","つける","聞く","よぶ",
                  "かれる","つまり","上","下","次","つぎ",
                  "わが国","自分","人々","人びと","別","他","間","話","例","形","日","家","手","名","身",
                  "そのもの","一つ","あと",

                  #2016/01/24 更に偏在度の高いものと、忘れてたひらがなを追加
                  "きゃ","きゅ","きょ","しゃ","しゅ","しょ","ちゃ","ちゅ","ちょ","にゃ","にゅ","にょ",
                  "ひゃ","ひゅ","ひょ","みゃ","みゅ","みょ","りゃ","りゅ","りょ","ゎ",
                  "事","目","とこ","中","字","お前","全部","きみ","もらう",
                  ]

    for h in hira:
        stopwords.add(h)
    for k in kata:
        stopwords.add(k)
    for h in hankata:
        stopwords.add(h)
    for k in kazu:
        stopwords.add(k)
    for z in zenkazu:
        stopwords.add(z)
    for k in kazukan:
        stopwords.add(k)
    for m in minialpha:
        stopwords.add(m)
    for b in bigalpha:
        stopwords.add(b)
    for h in han_minialpha:
        stopwords.add(h)
    for h in han_bigalpha:
        stopwords.add(h)
    for h in hiramoji:
        stopwords.add(h)
    for k in katamoji:
        stopwords.add(k)
    for h in han_katamoji:
        stopwords.add(h)
    for k in kanmoji:
        stopwords.add(unicode(k))
    for k in kigou:
        stopwords.add(k)
    for k in kokubu_words:
        stopwords.add(unicode(k))
    print "set([",
    for s in sorted(stopwords):
        print "u\"{0}\",".format(s),
    print "])"
