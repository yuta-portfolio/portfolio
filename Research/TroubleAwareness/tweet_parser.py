#!/usr/bin/env python
# coding=utf-8

u"Twitter関連のモジュール"

import sys
import time
import twitter
from urllib2 import URLError
from httplib import BadStatusLine
import MeCab
from collections import Counter
from function_common import load_from_mongo,delete_mongo,save_to_mongo,YmdHMS,pp,dprint,reservoir_sampling,extract_time,flatten_with_any_depth,get_stopwords
import codecs
import config
import re
import mojimoji
from nltk.util import ngrams
import datetime


__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__ = 'GPL'


class TaharaWord(object):
    u"""
    田原アルゴリズム用に単語を格納するクラス
    """
    def __init__(self,name):
        u"""初期化する
        Args:
            name:単語名で初期化する
        """
        self.name=name
        self.count=0
        self.days=set()

    def set_counter(self,created_at):
        u"""呟かれた日数をカウントする
        Args:
            created_at:呟かれた日
        """
        self.count+=1
        day=extract_time(created_at,"Y")+":"+extract_time(created_at,"M")+":"+extract_time(created_at,"D")
        self.days.add(day)


class TaharaUser(object):
    u"""
    田原アルゴリズム用のユーザーを格納するクラス
    """
    def __init__(self,name):
        u"""初期化する関数
        Args:
            name:ユーザー名
        """
        self.name=name
        self.words_name=set()
        self.words=[]

    def convert_words(self,ngrams,created_at):
        u"""ngram文をリスト化し、数えやすい形に変換

        Args:
            ngrams:Ngram文
            created_at:呟かれた日
        """
        for word in ngrams:
            if word not in self.words_name:
                self.words_name.add(word)
                t=TaharaWord(word)
                t.set_counter(word,created_at)
                self.words.append(t)
            else:
                t=[t for t in self.words if t.name == word]
                if len(t)>0:
                    t[0].set_counter(word,created_at)


class TaharaUsers(object):
    u"""
    田原アルゴリズム用複数のユーザー操作用クラス
    """
    def __init__(self):
        u"""初期化する

        """
        self.user_names=set()
        self.tehara_users=[]

    def convert_ngrams(self,user,created_at,ngrams):
        u"""ツイートをカウントしやすいように操作
        Args:
            user:ユーザー名
            created_at:ツイート作成日時
            ngrams:Ngram文

        """
        if user not in self.user_names:
            print "convert ngrams first"
            print len(self.tehara_users)
            self.user_names.add(user)
            t=TaharaUser(user)
            t.convert_words(ngrams,created_at)
            self.tehara_users.append(t)
        else:
            print "convert ngrams second.."
            t=[t for t in self.tehara_users if t.name == user]
            if(len(t)>0):
                t[0].convert_words(ngrams,created_at)
            else:
                print "Error! User is not found"

    def save_users(self,test=False):
        u"""ユーザーのデータを保存する
        Args:
            test:テストデータフラグ

        """
        print len(self.tehara_users)

        for user in self.tehara_users:
            print user.name
            words=user.words
            t_words=[]
            for w in words:
                t_words.append({'name':w.name,'count':w.count,'days':len(w.days)})

            if test==True:
                save_to_mongo({'_id':user.name,'words':t_words},"db","users_of_tehara_test")
            else:
                save_to_mongo({'_id':user.name,'words':t_words},"db","users_of_tehara")


g_tus=TaharaUsers()


class TwitterToken(object):
    u"""
    ツイート操作用クラス
    """
    def __init__(self,tweet):
        u"""初期化する
        Args:
            tweet:初期化するツイート
        """
        self.id=tweet["_id"]
        self.screen_name=tweet["user"]["screen_name"]
        self.text=self.filter(tweet["text"])

        ######################中退日時に合わせてcreated_atを改変#####################
        default_day=datetime.date(2015,1,1)
        yc=extract_time(tweet["created_at"],"Y")
        mc=extract_time(tweet["created_at"],"M")
        dc=extract_time(tweet["created_at"],"D")
        created_day=datetime.date(int(yc),int(mc),int(dc))

        profile=load_from_mongo("db","profile_of_dropout",criteria={"_id":self.screen_name})
        if len(profile)==0:
            profile=load_from_mongo("db","profile_of_normal",criteria={"_id":self.screen_name})
            if len(profile)==0:
                profile=load_from_mongo("db","profile_of_dropout_test",criteria={"_id":self.screen_name})
                if len(profile)==0:
                    profile=load_from_mongo("db","profile_of_normal_test",criteria={"_id":self.screen_name})
                    if len(profile)==0:
                        print "Error!! screen_name:",self.screen_name," is not exist in any list."

        start_day=profile[0]["date_of_start"]
        ys=extract_time(start_day,"Y")
        ms=extract_time(start_day,"M")
        ds=extract_time(start_day,"D")
        started_day=datetime.date(int(ys),int(ms),int(ds))
        day=default_day+(created_day-started_day)
        dummy_created_at="{0}:{1:0>2}:{2:0>2}:00:00:00".format(day.year,day.month,day.day)
        self.created_at=dummy_created_at

        ##########################################################################
        #改変なしver
        #self.created_at=tweet["created_at"]
        ##########################################################################

        self.wakachigaki=""
        self.ngrams=[]

    def filter(self,text):
        u"""ツイートからメタデータを削除する
        Args:
            text:ツイートのテキスト
        Returns:
            text:整形後のテキスト
        """
        # メンションを削除
        while "@" in text and " " in text:
            text=text.split(" ",text.count("@"))[-1]
        # タグを削除
        while "#" in text:
            text=text.split("#",1)[0]
        # URLを削除
        while "http" in text:
            text=text.split("http",1)[0]

        return text

    def wakachi(self):
        u"""分かち書きを行う

        Returns:
            辞書型で結果を返す
        """
        md=config.m_mecab_dic

        tagger=MeCab.Tagger(md.option)
        tagger.parse('')


        emoji=re.compile(u'^U00')
        kigou=re.compile(u'^[!-~]$')

        # 全角半角を正規化
        self.text=mojimoji.zen_to_han(self.text,kana=False,digit=True,ascii=True)
        self.text=mojimoji.han_to_zen(self.text,kana=True,digit=False,ascii=False)

        node=tagger.parseToNode(self.text.encode('utf-8'))
        words=[]

        while node:
            pos=node.feature.split(",")[md.pos]
            if pos=="形容詞" or pos == "形容動詞" or pos=="動詞" or pos=="名詞":
                if len(node.feature.split(","))<=md.base:
                    base = node.surface
                else:
                    base=node.feature.split(",")[md.base]

                if base == "*":
                    base = node.surface
                # 絵文字、ひらがな、カタカナ一文字は除外
                if (emoji.match(unicode(base)) is not None) or (kigou.match(unicode(base)) is not None):
                    pass
                # ストップワードに含まれたら除外
                elif unicode(base) in get_stopwords():
                    pass
                else:
                    # 大文字は小文字化して格納する
                    words.append(base.lower())
            node=node.next

        wakachi=map(str,words)
        wakachi = " ".join(wakachi)

        if "\n" in wakachi:
            wakachi=wakachi.split("\n",1)[0].strip()
        self.wakachigaki=wakachi

        return {'_id':self.id,'screen_name':self.screen_name,'text':self.text,'wakachi':wakachi}

    def ngrams_lists(self,start,end):
        u"""分かち書きをNgram化する

        分かち書きした後に使うこと

        Args:
            start:N >= start
            end:N <= end
        Returns:
            ngrams:Ngram文
        """
        if self.wakachigaki == "":
            return
        text=self.wakachigaki
        ret=[]

        #endがテキストより長いなら、テキストにに合わせる
        if len(text.split()) < end:
            end = len(text.split())

        for i in range(start,end):
            n_grams=ngrams(text.split(),i)
            terms=["".join(term) for term in n_grams]
            ret.append(terms)

        self.ngrams=(flatten_with_any_depth(ret))
        g_tus.convert_ngrams(self.screen_name,self.created_at,self.ngrams)

        return self.ngrams

    #分かち書きした後に呼び出すこと
    def ngrams_lists_clean(self,start,end):
        u"""分かち書きをNgram化する

        分かち書きの後に呼び出す

        Args:
            start:N >= start
            end:N <= end
        Returns:
            ngrams:Ngram文
        """
        if self.wakachigaki == "":
            return
        text=self.wakachigaki
        ret=[]
        #endがテキストより長いなら、テキストにに合わせる
        if len(text.split()) < end:
            end = len(text.split())

        for i in range(start,end):
            n_grams=ngrams(text.split(),i)
            terms=["".join(term) for term in n_grams]
            ret.append(terms)
        ret_cl=list(set(flatten_with_any_depth(ret)))

        return ret_cl


class TwitterTokenList(list):
    u"""
    TwitterTokenクラスを扱うクラス
    """
    def __init__(self):
        u"""初期化する
        """
        list.__init__(self)

    def count_frequency(self):
        u"""単語の頻度をカウントする

        Returns:
            単語のカウンター
        """
        words=[]
        md=config.m_mecab_dic
        for tweet in self:

            text=tweet.text

            tagger=MeCab.Tagger(md.option)

            # これないと、解析対象の文字列がPythonの管理下から外れてGCされる
            tagger.parse('')

            node=tagger.parseToNode(text.encode('utf-8'))
            while node:
                pos=node.feature.split(",")[md.pos]
                if pos=="名詞" or pos=="動詞":
                    #原型を格納する
                    if len(node.feature.split(","))<=md.base:
                        base = node.surface
                    else:
                        base=node.feature.split(",")[md.base]
                    if base == "*":
                        base = node.surface
                    words.append(base)
                node=node.next
        return Counter(words)

    def count_tweet_day_user(self,word):
        u"""検索ワードの使用者,使用日時を返す
        Args:
            word:検索ワード
        Returns:
            users:ユーザーのリスト
            times:ツイート作成日時のリスト
        """
        users=set()
        times=set()
        if word is None:
            return None,None

        for tweet in self:
            if word in tweet.ngrams:
                users.add(tweet.screen_name)
                day=extract_time(tweet.created_at,"Y")+":"+extract_time(tweet.created_at,"M")+":"+extract_time(tweet.created_at,"D")
                times.add(day)

        return users,times


class TwitterOperator(object):
    u"""
    Twitter関連の操作を行うインターフェース
    """
    def __init__(self,user=None,human=25,limit=200):
        u"""初期化する
        Args:
            user:ユーザー群の識別子
            human:人数制限
            limit:ツイート数制限
        """
        self.user=user
        self.human=human
        self.limit=limit
        self.teets=TwitterTokenList()
        count=0
        count_d=0
        count_n=0
        count_dt=0
        twts=TwitterTokenList()
        text_discontent=[]

        # 中退群のツイート
        if self.user.classify is "dropout":
            profiles=load_from_mongo(self.user.db,self.user.profile_coll)
            for profile in profiles:
                tweets=load_from_mongo(config.m_dropout.db,config.m_dropout.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                if len(tweets)==0:
                    tweets=load_from_mongo(config.m_dropout.db,"tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                    if len(tweets)==0:
                        continue
                for tweet in tweets:
                    twts.append(TwitterToken(tweet))

        # 普通の学生のツイート
        elif self.user.classify is "normal":
            profiles=load_from_mongo(self.user.db,self.user.profile_coll)
            for profile in profiles:
                tweets=load_from_mongo(config.m_normal.db,config.m_normal.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                if len(tweets)==0:
                    continue
                for tweet in tweets:
                    twts.append(TwitterToken(tweet))

        # 中退群のテスト用ツイート
        elif self.user.classify is "dropout_test":
            profiles=load_from_mongo(self.user.db,self.user.profile_coll)
            for profile in profiles:
                tweets=load_from_mongo(config.m_dropout_test.db,config.m_dropout_test.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                if len(tweets)==0:
                    tweets=load_from_mongo(config.m_dropout_test.db,"tweets_of_dropout?",criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                    if len(tweets)==0:
                        continue
                for tweet in tweets:
                    twts.append(TwitterToken(tweet))

        # 普通の学生のテスト用ツイート
        elif self.user.classify is "normal_test":
            profiles=load_from_mongo(self.user.db,self.user.profile_coll)
            for profile in profiles:
                tweets=load_from_mongo(config.m_normal_test.db,config.m_normal_test.tweets_coll,criteria={'$and':[{"user.screen_name":profile["_id"]},{'text':{'$regex':'^[^RT]'}},{'created_at':{'$gte':profile["date_of_start"],'$lte':profile["date_of_dropout"]}}]})
                if len(tweets)==0:
                    continue
                for tweet in tweets:
                    twts.append(TwitterToken(tweet))

        # 不満足ツイート
        elif self.user.classify is "discontent":
            texts=[]
            dprint("in discontent")
            #中退した学生の不満足ツイート
            with codecs.open("read/discontent.txt","r","utf-8") as f:
                lines=f.read()
                lines_sp=lines.split("----------\n")
                text_discontent=[text.lstrip().rstrip('\n') for text in lines_sp]
                dprint("codecs open")
                dprint(lines_sp)
                #同じ文書探してきて、IDラベル付
                for text in text_discontent:
                    if text == "\n" or text == " " or text == "　":
                        continue
                    tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":text})
                    if len(tweet) == 0:
                        tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":text})
                        if len(tweet)==0:
                            tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":text})
                            if len(tweet) == 0:
                                #全体一致が無ければ部分一致で探す
                                regex=r"^.*({0})$".format(text)
                                tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":{"$regex":regex}})
                                if len(tweet) == 0:
                                    tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":{"$regex":regex}})
                                    if len(tweet) == 0:
                                        tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":{"$regex":regex}})
                                        if len(tweet) == 0:
                                            #見つからない時は、仮のIDをふって格納する
                                            dprint([text,"Not found."])
                                            screen_name={"screen_name":"NotFound"}
                                            twts.append(TwitterToken({"_id":count,"text":text,"user":screen_name}))
                                            count+=1
                                            continue
                                        else:
                                            count_dt+=1
                                    else:
                                        count_n+=1
                                else:
                                    count_d+=1
                            else:
                                count_dt+=1
                        else:
                            count_n+=1
                    else:
                        count_d+=1
                    twts.append(TwitterToken(tweet[0]))

                print "Dropout:",count_d
                print "Normal:",count_n
                print "DropoutTest:{0}".format(str(count_dt))

        # 普通のツイートの処理
        elif self.user.classify is "content":
            texts=[]
            dprint("in content")
            #中退した学生の不満足ツイート
            with codecs.open("read/content.txt","r","utf-8") as f:
                lines=f.read()
                #長いけどこれが区切り文字
                lines_sp=lines.split("----------\n")
                text_discontent=[text.lstrip().rstrip('\n') for text in lines_sp]
                dprint(len(text_discontent))

                from config import m_discontent as dis
                k=len(load_from_mongo(dis.db,dis.wakachi_coll))
                #NANが多くなるため、多めに取得しておく→次元削減やめたので中止
                text_content=reservoir_sampling(text_discontent,k)

                dprint(len(text_content))
                #同じ文書探してきて、IDラベル付
                for text in text_content:
                    if text == "\n" or text == " " or text == "　":
                        continue
                    tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":text})
                    if len(tweet) == 0:
                        tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":text})
                        if len(tweet)==0:
                            tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":text})
                            if len(tweet) == 0:
                                #全体一致が無ければ部分一致で探す
                                regex=r"^.*({0})$".format(text)
                                tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":{"$regex":regex}})
                                if len(tweet) == 0:
                                    tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":{"$regex":regex}})
                                    if len(tweet) == 0:
                                        tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":{"$regex":regex}})
                                        if len(tweet) == 0:
                                            #見つからない時は、仮のIDをふって格納する
                                            dprint([text,"Not found."])
                                            screen_name={"screen_name":"NotFound"}
                                            twts.append(TwitterToken({"_id":count,"text":text,"user":screen_name}))
                                            count+=1
                                            continue
                                        else:
                                            count_dt+=1
                                    else:
                                        count_n+=1
                                else:
                                    count_d+=1
                            else:
                                count_dt+=1
                        else:
                            count_n+=1
                    else:
                        count_d+=1
                    twts.append(TwitterToken(tweet[0]))

                print "Dropout:",count_d
                print "Normal:",count_n

        elif self.user.classify is "discontent_test":
            texts=[]
            dprint("in discontent test")
            #中退した学生の不満足ツイート
            with codecs.open("read/discontent_test.txt","r","utf-8") as f:
                lines=f.read()
                lines_sp=lines.split("----------\n")
                text_discontent=[text.lstrip().rstrip('\n') for text in lines_sp]
                dprint("codecs open")
                dprint(lines_sp)
                #同じ文書探してきて、IDラベル付
                for text in text_discontent:
                    if text == "\n" or text == " " or text == "　" or text=="":
                        continue
                    tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":text})
                    if len(tweet) == 0:
                        tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":text})
                        if len(tweet)==0:
                            tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":text})
                            if len(tweet) == 0:
                                #全体一致が無ければ部分一致で探す
                                regex=r"^.*({0})$".format(text)
                                tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":{"$regex":regex}})
                                if len(tweet) == 0:
                                    tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":{"$regex":regex}})
                                    if len(tweet) == 0:
                                        tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":{"$regex":regex}})
                                        if len(tweet) == 0:
                                            #見つからない時は、仮のIDをふって格納する
                                            dprint([text,"Not found."])
                                            screen_name={"screen_name":"NotFound"}
                                            twts.append(TwitterToken({"_id":count,"text":text,"user":screen_name}))
                                            count+=1
                                            continue
                                        else:
                                            count_dt+=1
                                    else:
                                        count_n+=1
                                else:
                                    count_d+=1
                            else:
                                count_dt+=1
                        else:
                            count_n+=1
                    else:
                        count_d+=1
                    twts.append(TwitterToken(tweet[0]))

                print "Dropout:",count_d
                print "Normal:",count_n
                print "DropoutTest:{0}".format(str(count_dt))
            pass
        elif self.user.classify is "content_test":
            texts=[]
            dprint("in content test")
            #中退した学生の不満足ツイート
            with codecs.open("read/content.txt","r","utf-8") as f:
                lines=f.read()
                #長いけどこれが区切り文字
                lines_sp=lines.split("----------\n")
                text_discontent=[text.lstrip().rstrip('\n') for text in lines_sp]
                dprint(len(text_discontent))
                from config import m_discontent_test as dis
                k=len(load_from_mongo(dis.db,dis.wakachi_coll))
                #NANが多くなるため、多めに取得しておく→中止
                text_content=reservoir_sampling(text_discontent,k)

                dprint(len(text_content))
                #同じ文書探してきて、IDラベル付
                for text in text_content:
                    if text == "\n" or text == " " or text == "　":
                        continue
                    tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":text})
                    if len(tweet) == 0:
                        tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":text})
                        if len(tweet)==0:
                            tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":text})
                            if len(tweet) == 0:
                                #全体一致が無ければ部分一致で探す
                                regex=r"^.*({0})$".format(text)
                                tweet=load_from_mongo(self.user.db,"tweets_of_dropout",criteria={"text":{"$regex":regex}})
                                if len(tweet) == 0:
                                    tweet=load_from_mongo(self.user.db,"tweets_of_normal",criteria={"text":{"$regex":regex}})
                                    if len(tweet) == 0:
                                        tweet=load_from_mongo(self.user.db,"tweets_of_dropout?",criteria={"text":{"$regex":regex}})
                                        if len(tweet) == 0:
                                            #見つからない時は、仮のIDをふって格納する
                                            dprint([text,"Not found."])
                                            screen_name={"screen_name":"NotFound"}
                                            twts.append(TwitterToken({"_id":count,"text":text,"user":screen_name}))
                                            count+=1
                                            continue
                                        else:
                                            count_dt+=1
                                    else:
                                        count_n+=1
                                else:
                                    count_d+=1
                            else:
                                count_dt+=1
                        else:
                            count_n+=1
                    else:
                        count_d+=1
                    twts.append(TwitterToken(tweet[0]))
                    if len(twts) >= k:
                        break
                print "Dropout:",count_d
                print "Normal:",count_n

        else:
            print "classify {0} is not Found".format(self.user.classify)
            return

        if len(twts) > len(text_discontent):
            dprint(len(twts))
            dprint("Size of tweet array is over the base text.")

        self.tweets=twts

    def save_wakachi(self):
        u"""関数の説明
        Args:
            hoge:引数の説明
        Returns:
            foo:戻り値の説明
        """
        delete_mongo(self.user.db,self.user.wakachi_coll)
        for t in self.tweets:
            save_to_mongo(t.wakachi(),self.user.db,self.user.wakachi_coll)

    def save_tahara_format(self):
        u"""田原アルゴリズム使用の前処理を行う

        """
        counter=Counter()
        loop_counter=0

        # コレクションをリセット
        delete_mongo(self.user.db,self.user.tahara_coll)

        #全ツイートを分かち書き後、1-7gram化
        for tweet in self.tweets:
            print "(",loop_counter+1,"/",len(self.tweets),")",u"Now Parsing……:",pp(tweet.text)
            tweet.wakachi()
            terms=tweet.ngrams_lists(1,7)
            loop_counter+=1
            if terms is None:
                continue
            counter+=Counter(terms)

        # Ngram化するときに保存しておいた重複なしのユーザー情報をDBのUsers-of-Teharaに保存
        # 保存するのはユーザー名、作成日時、N-grams
        g_tus.save_users()

        # 使われたワードを降順でループ
        for word,cnt in counter.most_common():
            #一回しか使われてないワードはカット
            if cnt <= 1:
                print word," is used only once."
                continue
            print "Count Words Of ",cnt," : ",self.user.classify,":",word

            # そのワードを呟いたユーザー数と、期間中何日呟かれたかをDBに保存する
            user,days=self.tweets.count_tweet_day_user(word)
            if user is None:
                continue
            save_to_mongo({"_id":word,"count":cnt,'tweet_users':len(user),'tweet_days':len(days)},self.user.db,self.user.tahara_coll)

    def save_tahara_format_clean(self):
        u"""田原アルゴリズム前処理。上と保存形式を変更
        """
        counter=Counter()
        loop_counter=0
        # dbをクリーンアップ
        delete_mongo(self.user.db,self.user.tahara_coll_clean)
        #全ツイートを分かち書き後、1-7gram化
        for tweet in self.tweets:
            print "(",loop_counter+1,"/",len(self.tweets),")",u"Now Parsing……:",pp(tweet.text)
            tweet.wakachi()
            print("end wakachi")
            terms=tweet.ngrams_lists_clean(1,7)
            print pp(terms)
            print ("end_Ngram")
            loop_counter+=1
            if terms is None:
                continue
            counter+=Counter(terms)

        # 使われたワードを降順でループ
        for word,cnt in counter.most_common():
            #一回しか使われてないワードはカット
            if cnt <= 1:
                print word," is used only once."
                continue
            print "Count Words Of ",cnt," : ",self.user.classify,":",word
            save_to_mongo({"_id":word,"include_tweet":cnt},self.user.db,self.user.tahara_coll_clean)


def make_twitter_request(twitter_api_func,max_errors=10,*args,**kw):
    u"""頑強なTwitterリクエストを発行する

    Args:
        twitter_api_func:TwitterAPI+
        max_errors:最大エラー数
    """
    def handle_twitter_http_error(e,wait_period=2,sleep_when_rate_limited=True):
        u"""よくあるHTTPエラーを処理するネストされたヘルパー関数

            問題が500台のエラーなら、更新したwait_periodを返す
            アクセス回数制限の問題(429エラー)ならアクセス回数制限がリセットされるまでブロックする
            呼び出し元で特別な処理をしなければならない401、404エラーにはNONEを返す
        Args:
            e:エラー
            wait_period:wait秒
            sleep_when_rate_limited:API制限に引っかかった時15分待つか
        """
        if wait_period > 3600:#Seconds
            print >>sys.stderr,'Too many retries.Quitting.'
            raise e

        # よくあるエラーコードについては
        # https://dev.twitter.com/docs/error-codes-responsedを参照

        if e.e.code==401:
            print >> sys.stderr,'Encountered 401 Error(Not Authorized)'
            return None
        elif e.e.code==404:
            print >> sys.stderr,'Encountered 404 Error(Not Found)'
            return None
        elif e.e.code==429:
            print >> sys.stderr,'Encountered 429 Error(Rate Limit Exceeded)'
            if sleep_when_rate_limited:
                print >> sys.stderr,"Retrying in 15 minutes...ZzZ..."
                sys.stderr.flush()
                time.sleep(60*15+5)
                print >> sys.stderr,'...ZzZ...Awake now and trying again.'
                return 2
            else:
                raise e # アクセス回数制限の問題は呼び出し元で処理しなければならない
        elif e.e.code in (500,502,503,504):
            print >> sys.stderr,'Encountered %i Error.Retrying in %i seconds'%\
                                (e.e.code,wait_period)
            time.sleep(wait_period)
            wait_period*=1.5
            return wait_period
        else:
            raise e
    # ネストされたヘルパーの関数の末尾

    wait_period=2
    error_count=0

    while True:
        try :
            return twitter_api_func(*args,**kw)
        except twitter.api.TwitterHTTPError,e:
            error_count=0
            wait_period=handle_twitter_http_error(e,wait_period)
            if wait_period is None:
                return
        except URLError,e:
            error_count+=1
            print >> sys.stderr,'URLError encountered. Continuing.'
            if error_count > max_errors:
                print >> sys.stderr,"Too many consecutive errors...bailing out."
                raise
        except BadStatusLine,e:
            error_count +=1
            print >> sys.stderr,"BadStatusLine encountered.Continuing."
            if error_count > max_errors:
                print >> sys.stderr,"Too many consecutive errors...bailing out."
                raise

def harvest_user_timeline(twitter_api,screen_name=None,user_id=None,max_results=1000):
    u"""ユーザーのタイムラインを取得する

        Args:
            twitter_api:TwitterAPI
            screen_name:ユーザー名
            user_id:ユーザーID
            max_results:最大取得ツイート数
        Returns:
            取得したツイート
        """
    assert(screen_name!=None)!=(user_id!=None),\
    "Must have screen_name or user_id, but not both"

    kw={ # Keyword args for the Twitter API call
        'count':200,
        'trim_user':'true',
        'include_rts':'true',
        'since_id':1
        }

    if screen_name:
        kw['screen_name']=screen_name
    else:
        kw['user_id']=user_id

    max_pages=16
    results=[]

    tweets = make_twitter_request(twitter_api.statuses.user_timeline,**kw)

    if tweets is None:# 401(Not Authorized)-ループ処理に備えておく
        tweets=[]

    results+=tweets

    print >> sys.stderr,'Fetched %i tweets' %len(tweets)

    page_num=1

    if max_results==kw['count']:
        page_num = max_pages  # ループに入るのを防ぐ

    while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:
        kw['max_id']=min([tweet['id'] for tweet in tweets]) -1
        tweets = make_twitter_request(twitter_api.statuses.user_timeline,**kw)
        results += tweets

        print >> sys.stderr,'Fetched %i tweets'%(len(tweets),)

        page_num+=1

    print >> sys.stderr,'Done fetching tweets'

    return results[:max_results]


def save_tweet(data,mongo_db,mongo_db_coll,screen_name="",**mongo_conn_kw):
    u"""ツイートを保存する
        Args:
            data:保存するデータ
            mongo_db:保存するデータベース
            mongo_db_coll:保存するコレクション名
    """
    for tweet in data:
        tweet_id = tweet['id']
        created_at=YmdHMS(tweet['created_at'])
        text=tweet['text'].encode('utf-8')

        user_mentions=[{'screen_name':user_mention['screen_name']}
                  for user_mention in tweet['entities']['user_mentions']]
        urls = [{'expanded_url':url['expanded_url']}
                    for url in tweet['entities']['urls']]
        hashtags=[{'text':hashtag['text']}
                  for hashtag in tweet['entities']['hashtags']]
        media=[]
        entities={'hashtags':hashtags,'media':media,'urls':urls,'user_mentions':user_mentions}
        retweet_count=tweet['retweet_count']
        favorite_count=tweet['favorite_count']
        coordinates=tweet["coordinates"]
        source=tweet["source"]
        user_id=tweet["user"]["id"]
        user={"id":user_id,'screen_name':screen_name}
        save_to_mongo({'_id':tweet_id,'created_at':created_at,'text':text,'entities':entities,\
                     'coordinates':coordinates,'retweet_count':retweet_count,'favorite_count':favorite_count,\
                     'user':user,'source':source},mongo_db,mongo_db_coll)


def get_twitter_api():
    u"""TwitterAPIを取得する
        Returns:
            TwitterAPI
    """
    auth=twitter.oauth.OAuth(config.ACCESS_KEY,config.ACCESS_SECRET,config.CONSUMER_KEY,config.CONSUMER_SECRET)
    return twitter.Twitter(auth=auth)


def get_members_from_list(screen_name,list_name):
    u"""リストからメンバーを取得する
        Args:
            screen_name:リスト保持者
            list_name:取得するリスト名
        Returns:
            メンバーのリスト
    """
    twitter_api=get_twitter_api()
    member_list_json=twitter_api.lists.list(screen_name=screen_name)
    list_id=0

    #list_nameからlist_idを探す
    for m in member_list_json:
        if m['name'] == list_name:
            list_id = m['id']

    #リストが存在しなければエラーを出して返す
    if list_id == 0:
        print "Not Exist:Such a list_name "+list_name
        return

    members=twitter_api.lists.members(list_id=list_id,count=100)
    #メンバーをリストにして返す
    member_list = [member['screen_name']
                        for member in members['users']]
    return member_list


def update_tweet_from_list(coll_name,user_name,list_name):
    u"""リストに含まれるユーザーのツイートを更新
        Args:
            coll_name:コレクション名
            user_name:ユーザー名
            list_name:リスト名
    """
    twitter_api=get_twitter_api()
    member_list=get_members_from_list(user_name,list_name)
    for member in member_list:
        search_results=harvest_user_timeline(twitter_api,screen_name=member,max_results=3200)
        #何故かharvestにしてからscreen_nameが取れなくなったので、名前を引数で追加
        save_tweet(search_results,'db',coll_name,screen_name=member)


def update_tweet_from_names(coll_name,names=[]):
    u"""スクリーンネームからツイートを更新
        Args:
            coll_name:コレクション名
            names:ユーザー名
    """
    from tqdm import tqdm
    twitter_api=get_twitter_api()
    for name in tqdm(names):
        try:
            search_results=harvest_user_timeline(twitter_api,screen_name=name,max_results=3200)
            save_tweet(search_results,'db',coll_name,screen_name=name)
        except:
            print "AssertionError"

def update_tweet():
    u"""ツイートを更新する
    """
    update_tweet_from_list("tweets_of_normal",config.MASTER_SCREEN_NAME,"normal_extend")