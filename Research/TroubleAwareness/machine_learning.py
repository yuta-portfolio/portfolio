#!/usr/bin/env python
# coding=utf-8

u"機械学習に関連するモジュール"
from functools import reduce

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score, f1_score
from function_common import load_from_mongo,pp,dprint,write_to_text,save_to_mongo,create_indexes, distance
import config
from sklearn.metrics import precision_recall_curve,roc_curve,auc,f1_score
from sklearn.cross_validation import ShuffleSplit, LeaveOneOut
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
from data_factory import load_stopwords_old,create_ngram_model_rf,make_document_corpus,create_ngram_model_svm
import tweet_parser as tp


__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__ = 'GPL'


def random_forest(train_x,train_y,test_x,test_y):
    u"""ランダムフォレストのテスト

        Args:
            train_x:訓練データ特徴
            train_y:訓練データラベル
            test_x:テストデータ特徴
            test_y:テストデータラベル

    """
    forest=RandomForestClassifier(n_estimators=100)
    forest=forest.fit(train_x,train_y)
    output=forest.predict(train_x)

    print output

    print "正解率:",accuracy_score(test_y,output)
    print "f1:",f1_score(test_y,output)
    print "適合率:",precision_score(test_y,output)
    print "再現率:",recall_score(test_y,output)
    print "ROC:",roc_auc_score(test_y,output)


def train_model(clf_factory,X,Y):
    u"""モデルを訓練する

        Args:
            clf_factory:訓練するモデル
            X:特徴量
            Y:正解ラベル
    """
    cv=ShuffleSplit(n=len(X),n_iter=10,test_size=0.3,random_state=0)
    train_errors=[]
    test_errors=[]
    scores=[]
    pr_scores=[]
    precisions,recalls,thresholds=[],[],[]

    for train,test in cv:
        X_train,y_train=X[train],Y[train]
        X_test,y_test=X[test],Y[test]
        clf=clf_factory
        clf.fit(X_train,y_train)
        train_score=clf.score(X_train,y_train)
        test_score=clf.score(X_test,y_test)
        train_errors.append(1-train_score)
        test_errors.append(1-test_score)
        scores.append(test_score)
        proba=clf.predict_proba(X_test)
        fpr,tpr,roc_thresholds=roc_curve(y_test,proba[:,1])
        precision,recall,pr_thresholds=precision_recall_curve(y_test,proba[:,1])
        pr_scores.append(auc(recall,precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    scores_to_sort=pr_scores
    median=np.argsort(scores_to_sort)[len(scores_to_sort)/2]
    summary = (np.mean(scores), np.std(scores), np.mean(pr_scores), np.std(pr_scores))
    print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary


def sk_grid_search(model,X,Y,param_grid,save_file):
    u"""最適なパラメータを調べる
        Args:
            model:機械学習モデル
            X:特徴量
            Y:ラベル
            param_grid:パラメータ
            save_file:保存するファイル名
        Returns:
            clf:最も良かったモデル
    """
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)
    grid_search = GridSearchCV(model(),
                              param_grid=param_grid,
                              cv=cv,
                              verbose=10,
                                n_jobs=-1
                               )
    grid_search.fit(X,Y)
    clf = grid_search.best_estimator_
    write_to_text(grid_search.best_params_,save_file)

    return clf


def grid_search_model(clf_factory, X, Y,save_file="read/best_param.txt"):
    u"""最適なパラメータを調べる
        Args:
            clf_factory:機械学習モデル
            X:特徴量
            Y:ラベル
        Returns:
            clf:最も良かったモデル
    """
    stopwords=load_stopwords_old()
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)
    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      vect__min_df=[1, 2],
                      vect__stop_words=[None, stopwords],
                      vect__smooth_idf=[False, True],
                      vect__use_idf=[False, True],
                      vect__sublinear_tf=[False, True],
                      vect__binary=[False, True],
                      )
    grid_search = GridSearchCV(clf_factory(),
                              param_grid=param_grid,
                              cv=cv,
                              verbose=10)
    grid_search.fit(X, Y)
    clf = grid_search.best_estimator_
    write_to_text(grid_search.best_params_,save_file)

    return clf


def get_best_model():
    u"""仮の最適なモデルを返す

        Returns:
            clf:仮の最適なモデル
    """
    best_params = dict(vect__ngram_range=(1, 2),
                      vect__min_df=1,
                      vect__stop_words=None,
                      vect__smooth_idf=False,
                      vect__use_idf=False,
                      vect__sublinear_tf=True,
                      vect__binary=False,
                      )
    best_clf = create_ngram_model_svm(best_params)

    return best_clf


def ml_test(pX,py,nX,ny,pX_t,py_t,nX_t,ny_t):
    u"""機械学習のテストをする
        Args:
            pX:posの特徴量
            py:posのラベル
            nX:negの特徴量
            ny:negのラベル
    """
    X=pX+nX
    y=py+ny
    npX=np.array(X)
    npY=np.array(y)

    npX_t=np.array(nX_t)

    forest=create_ngram_model_rf()
    forest=forest.fit(npX,npY)
    output=forest.predict(npX_t)
    output=pd.DataFrame(output)

    names=load_from_mongo("db","profile_of_normal_test")

    dprint("normal test")
    for profile in names:
        twts=tp.TwitterTokenList()
        tweets=load_from_mongo("db","tweets_of_normal",criteria={'$and':[{"user.screen_name":profile["_id"]},{"text":{'$regex':'^[^RT]'}}]},limit=200)
        for tweet in tweets:
            twts.append(tp.TwitterToken(tweet))
        wakachis=[w.wakachi() for w in twts]
        tweets=make_document_corpus(wakachis)
        try:
            output=forest.predict(tweets)
        except:
            pass
        print type(output)
        output=pd.DataFrame(output)
        print len(output[output == True].dropna())

    dprint("dropout test")
    names=load_from_mongo("db","profile_of_dropout_test")
    for profile in names:
        twts=tp.TwitterTokenList()
        tweets=load_from_mongo("db","tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{"text":{'$regex':'^[^RT]'}},{'created_at':{'$lte':profile["date_of_dropout"]}}]},limit=200)
        for tweet in tweets:
            twts.append(tp.TwitterToken(tweet))
        wakachis=[w.wakachi() for w in twts]
        tweets=make_document_corpus(wakachis)
        try:
            output=forest.predict(tweets)
        except:
            pass
        print type(output)
        output=pd.DataFrame(output)
        print len(output[output == True].dropna())


def tahara_algorithm():
    u"""田原アルゴリズムで特徴度を計算してソート

    """
    from decimal import Decimal

    # ここだけ弄る
    normal_man=30
    dropout_man=30
    dropout_all_days=100

    # 中退者の特徴度
    words_dropout=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll)
    words_normal=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll)

    for word in words_dropout:
        print "In Words of Dropout:",word["_id"]
        word_normal_list=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,criteria={"_id":word["_id"]})
        is_normal_tweet=1
        if len(word_normal_list)==0:
            word_normal={"_id":word["_id"],"count":0,"tweet_users":0,'tweet_days':0}
            is_normal_tweet=0
        else:
            word_normal=word_normal_list[0]

        t1=Decimal(word["count"])/((Decimal(1)/Decimal(2))*(Decimal(word["count"])+Decimal(word_normal["count"])))
        t2=Decimal(2)/(Decimal(1)+Decimal(is_normal_tweet))
        t3=Decimal(word["tweet_users"])/Decimal(dropout_man)
        t4=Decimal(word["tweet_days"])/Decimal(dropout_all_days)
        t5=Decimal(t1)*Decimal(t2)*Decimal(t3)*Decimal(t4)
        save_to_mongo({"_id":word["_id"],"count":word["count"],"tweet_days":word["tweet_days"],"tweet_users":word["tweet_users"],"rate":float(t5)},config.m_dropout.db,config.m_dropout.tahara_coll)

    for word2 in words_normal:
        print "In Words of Normal:",word2["_id"]
        word_dropout_list=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,criteria={"_id":word2["_id"]})
        is_dropout_tweet=1
        if len(word_dropout_list)==0:
            word_dropout={"_id":word2["_id"],"count":0,"tweet_users":0,'tweet_days':0}
            is_dropout_tweet=0
        else:
            word_dropout=word_dropout_list[0]

        t1=Decimal(word2["count"])/((Decimal(1)/Decimal(2))*(Decimal(word2["count"])+Decimal(word_dropout["count"])))
        t2=Decimal(2)/(Decimal(1)+Decimal(is_dropout_tweet))
        t3=Decimal(word2["tweet_users"])/Decimal(normal_man)
        t4=Decimal(word2["tweet_days"])/Decimal(dropout_all_days)
        t5=Decimal(t1)*Decimal(t2)*Decimal(t3)*Decimal(t4)
        save_to_mongo({"_id":word2["_id"],"count":word2["count"],"tweet_days":word2["tweet_days"],"tweet_users":word2["tweet_users"],"rate":float(t5)},config.m_normal.db,config.m_normal.tahara_coll)

    create_indexes(config.m_dropout.db,config.m_dropout.tahara_coll,["rate"])
    create_indexes(config.m_normal.db,config.m_normal.tahara_coll,["rate"])

    dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,True)
    normals=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,True)
    import pymongo
    count=0
    for d in dropouts.sort("rate",pymongo.DESCENDING):
        print pp(d)
        count+=1
        if count > 100:
            break
    count=0
    print "--------D vs N----------------"
    for n in normals.sort("rate",pymongo.DESCENDING):
        print pp(n)
        count+=1
        if count > 100:
            break


def tahara_algorithm_clean():
    u"""田原アルゴリズムで特徴度を計算してソート

    上記でミスがあったので修正

    """
    from decimal import Decimal

    # ここだけ弄る
    normal_man=30
    dropout_man=30
    dropout_all_days=100

    # 中退者の特徴度
    words_dropout=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll)
    words_normal=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll)

    for word in words_dropout:
        print "In Words of Dropout:",word["_id"]
        word_normal_list=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,criteria={"_id":word["_id"]})
        is_normal_tweet=1
        if len(word_normal_list)==0:
            word_normal={"_id":word["_id"],"count":0,"tweet_users":0,'tweet_days':0}
            is_normal_tweet=0
        else:
            word_normal=word_normal_list[0]

        temp=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll_clean,criteria={"_id":word["_id"]})
        print pp(temp)
        if len(temp)==0:
            count_dropout=0
        else:
            try:
                count_dropout=temp[0]["include_tweet"]
            except:
                continue
        temp=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll_clean,criteria={"_id":word["_id"]})
        print pp(temp)
        if len(temp)==0:
            count_normal=0
        else:
            try:
                count_normal=temp[0]["include_tweet"]
            except:
                continue
        if count_normal==0 and count_dropout ==0:
            continue

        t1=Decimal(count_dropout)/((Decimal(1.)/Decimal(2.))*(Decimal(count_dropout)+Decimal(count_normal)))
        t2=Decimal(2.)/(Decimal(1.)+Decimal(is_normal_tweet))
        t3=Decimal(word["tweet_users"])/Decimal(dropout_man)
        t4=Decimal(word["tweet_days"])/Decimal(dropout_all_days)
        t5=Decimal(t1)*Decimal(t2)*Decimal(t3)*Decimal(t4)
        print t5
        save_to_mongo({"_id":word["_id"],"include_tweet":count_dropout,"rate":float(t5)},config.m_dropout.db,config.m_dropout.tahara_coll_clean)

    for word2 in words_normal:
        print "In Words of Normal:",word2["_id"]
        word_dropout_list=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,criteria={"_id":word2["_id"]})
        is_dropout_tweet=1
        if len(word_dropout_list)==0:
            word_dropout={"_id":word2["_id"],"count":0,"tweet_users":0,'tweet_days':0}
            is_dropout_tweet=0
        else:
            word_dropout=word_dropout_list[0]

        temp=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll_clean,criteria={"_id":word["_id"]})
        if len(temp)==0:
            count_dropout=0
        else:
            count_dropout=temp[0]["include_tweet"]
        temp=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll_clean,criteria={"_id":word["_id"]})
        if len(temp)==0:
            count_normal=0
        else:
            count_normal=temp[0]["include_tweet"]
        if count_normal==0 and count_dropout ==0:
            continue

        t1=Decimal(count_normal)/((Decimal(1.)/Decimal(2.))*(Decimal(count_normal)+Decimal(count_dropout)))
        t2=Decimal(2.)/(Decimal(1.)+Decimal(is_dropout_tweet))
        t3=Decimal(word2["tweet_users"])/Decimal(normal_man)
        t4=Decimal(word2["tweet_days"])/Decimal(dropout_all_days)
        t5=Decimal(t1)*Decimal(t2)*Decimal(t3)*Decimal(t4)
        print t5
        save_to_mongo({"_id":word2["_id"],"include_tweet":count_normal,"rate":float(t5)},config.m_normal.db,config.m_normal.tahara_coll_clean)

    create_indexes(config.m_dropout.db,config.m_dropout.tahara_coll_clean,["rate"])
    create_indexes(config.m_normal.db,config.m_normal.tahara_coll_clean,["rate"])

    dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll_clean,True)
    normals=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll_clean,True)
    import pymongo
    count=0
    for d in dropouts.sort("rate",pymongo.DESCENDING):
        print pp(d)
        count+=1
        if count > 100:
            break
    count=0
    print "--------D vs N----------------"
    for n in normals.sort("rate",pymongo.DESCENDING):
        print pp(n)
        count+=1
        if count > 100:
            break


def sim_user(classify="dropout",is_classify="dropout"):
    u"""ユーザーのコサイン類似度を求める

        Args:
            classify:類似度を計算するユーザー群
            is_classify:どのユーザー群らしさを計算するか
    """
    import pymongo
    from decimal import Decimal
    import math
    from function_common import get_word_point
    sim_dropouts=[]
    if classify == "dropout":
        profiles=load_from_mongo(config.m_dropout.db,config.m_dropout.profile_coll)
    else:
        profiles=load_from_mongo(config.m_normal.db,config.m_normal.profile_coll)
    for profile in profiles:

        wui=load_from_mongo("db","users_of_tehara",return_cursor=False,criteria={'_id':profile['_id']})
        if (wui is None) or (len(wui)==0):
            continue
        words= wui[0][u'words']
        wui_wci=0
        w2ui=0
        w2ci=0
        count=0
        if is_classify =="dropout":
            dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,True)
        else:
            dropouts=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,True)
        for d in dropouts.sort("rate",pymongo.DESCENDING):
            count+=1
            word=[w for w in words if w["name"] == d["_id"]]
            if len(word)==0:
                word=[{"count":0,"name":d["_id"],"days":0}]
            else:
                pass

            wui_wci+=(Decimal(word[0]["count"])*Decimal(d["rate"]))*get_word_point(d["_id"])
            w2ui+=math.pow(Decimal(word[0]["count"]),2)
            w2ci+=math.pow(Decimal(d["rate"]),2)

            if count > 1000:
                break
        sim_dropout=Decimal(wui_wci)/(Decimal(math.sqrt(Decimal(w2ui)))*Decimal(math.sqrt(Decimal(w2ci))))
        sim_dropouts.append(sim_dropout)
        d=None
    print pp(sim_dropouts)


def nearest_neighbor(train, train_targets, point):
    u"""最近傍法
        Args:
            train:特徴ベクトル
            train_targets:特徴ベクトル
            point:特徴点
        Returns:
            最近防の点を返す
    """
    # point と train の各点のユークリッド距離を測る
    distances = np.array([distance(t, point) for t in train])
    # 距離が最小 (再近傍) の点を得る
    nearest_point = distances.argmin()
    # 再近傍の点の種別を判定結果として返す
    return train_targets[nearest_point]


def k_kinbo(m_train):
    u"""k近傍法
        Args:
            m_train:訓練データ

    """
    data_n=np.array(pd.read_csv(m_train,sep=','))[:,:]
    data_p=pd.read_csv(m_train,sep=',',header=0)

    df2=data_p.dropna()
    train_data=df2.values

    #学習させる
    xs=train_data[:,2:]
    y=train_data[:,1]

    #特徴量をZスコアに正規化する
    xs-=xs.mean(axis=0)
    xs/=xs.std(axis=0)

    #leave-one-out 交差検定
    loo=LeaveOneOut(len(xs))
    results=[]
    for train_indexes,test_indexes in loo:
        train=xs[train_indexes]
        test=xs[test_indexes]
        #k近傍で判定
        answer= nearest_neighbor(train, y, test)
        right=y[test_indexes]
        results.append(answer==right)
    N=len(xs)

    correct = reduce(lambda n, o: n + 1 if o else n, results, 0)
    msg = '正解: {0}/{1}'.format(correct, N)
    print(msg)
    failed = reduce(lambda n, o: n + 1 if not o else n, results, 0)
    msg = '不正解: {0}/{1}'.format(failed, N)
    print(msg)
    correct_rate = (float(correct) / N) * 100
    msg = '正解率: {0}%'.format(correct_rate)
    print(msg)


def classify_chushikoku():
    u"""中四国の学生に悩み事検出アルゴリズムを適用する

    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    data=pd.read_csv("no_taigaku_train4.csv",sep=",",header=0)
    train=data.values
    x=train[:,2:]
    y=train[:,1]

    data_t=pd.read_csv("no_taigaku_chushikoku.csv",sep=",",header=0)
    test=data_t.values
    x_test=test[:,1:]
    name_test=test[:,0]
    print name_test
    print len(x_test)

    best_forest=RandomForestClassifier(bootstrap=True,class_weight=None,criterion="gini",
                                  max_depth=5,max_features="sqrt",max_leaf_nodes=None,
                                  min_samples_leaf=1,min_samples_split=3,
                                  min_weight_fraction_leaf=0.001,n_estimators=30,n_jobs=-1,
                                  oob_score=False,random_state=None,verbose=0,warm_start=False)
    best_svm=SVC(C=100,kernel="linear",gamma=0.001)

    clf=best_svm.fit(x,y)
    label_predict=clf.predict(x_test)

    print name_test[label_predict=="chutai"]
    print len(name_test[label_predict=="chutai"])


def test_JSAI():
    u"""JSAI用

    """
    import pandas as pd
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    LOOP=10
    data=pd.read_csv("no_taigaku_train4.csv",sep=",",header=0)
    train=data.values
    x=train[:,2:]
    y=train[:,1]
    best_forest=RandomForestClassifier(bootstrap=True,class_weight=None,criterion="gini",
                                  max_depth=5,max_features="sqrt",max_leaf_nodes=None,
                                  min_samples_leaf=1,min_samples_split=3,
                                  min_weight_fraction_leaf=0.001,n_estimators=30,n_jobs=-1,
                                  oob_score=False,random_state=None,verbose=0,warm_start=False)

    cv=ShuffleSplit(n=len(x),n_iter=LOOP,test_size=0.1,random_state=None)
    nb=MultinomialNB
    param=dict(
        alpha=[0,0.01,0.1,0.5,1],
        fit_prior=[True,False],
    )
    best_nb=MultinomialNB(alpha=0.01,fit_prior=False)

    param=dict(
        kernel=['linear','rbf'],
        gamma=[1e-3,1e-4],
        C=[1,10,100,1000],
    )
    best_svc=SVC(kernel="linear",gamma=0.001,C=10)

    param=dict(
        n_estimators=[5,10,20,30,50,100],
        max_features=['auto','sqrt','log2'],
        min_samples_split=[3,5,10,20,30,50,100],
        max_depth=[None,3,5,10,20,30,50,100],
        bootstrap=[True,False],
        class_weight=[None,"balanced","balanced_subsample"],
        criterion=["gini","entropy"]

    )


def accuracy(tp=0.,tn=0.,fp=0.,fn=0.):
    u"""機械学習の精度を検証する
        Args:
            tp:TruePositive
            tn:TrueNegative
            fp:FalsePositive
            fn:FalseNegative
    """
    acc=(tp+tn)/(tp+fp+fn+tn)
    pre=tp/(tp+fp)
    rec=tp/(tp+fn)
    f=((2.*rec*pre)/(rec+pre))

    print u"正解率:",acc
    print u"適合率:",pre
    print u"再現率:",rec
    print u"F値:",f


def random_forest_from_csv(m_train, m_test):
    u"""CSVのデータにランダムフォレストを適用する
        Args:
            m_train:訓練用データ
            m_test:テスト用データ

    """
    data_p=pd.read_csv(m_train,sep=',',header=0)

    train_data=data_p.values

    print train_data

    #学習させる
    xs=train_data[:,2:]
    y=train_data[:,1]
    print xs
    print xs.shape
    print y
    print y.shape
    y=[1]*30
    y.extend([0]*30)

    forest=RandomForestClassifier(bootstrap=True,class_weight=None,criterion="gini",
                                  max_depth=10,max_features=20,max_leaf_nodes=None,
                                  min_samples_leaf=1,min_samples_split=3,
                                  min_weight_fraction_leaf=0.0,n_estimators=300,n_jobs=1,
                                  oob_score=False,random_state=0,verbose=0,warm_start=False)

    forest=forest.fit(xs,y)
    test=pd.read_csv(m_test,sep=',',header=0)
    test2=test.values
    xs_test=test2[:,2:]
    y_test=test2[:,1]
    y_test=[1]*7
    y_test.extend([0]*7)

    output=forest.predict(xs_test)

    print "正解率:",accuracy_score(y_test,output)
    print "f1:",f1_score(y_test,output)
    print "適合率:",precision_score(y_test,output)
    print "再現率:",recall_score(y_test,output)
    print "ROC:",roc_auc_score(y_test,output)

    print output