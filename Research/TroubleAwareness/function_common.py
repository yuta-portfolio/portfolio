#!/usr/bin/env python
# coding=utf-8

u"汎用的なメソッド集"

import math
import gensim
import numpy as np
import scipy as sp
import random
import pprint
import re
import calendar
import time
import pymongo
import sys
import codecs
import config
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from numba import jit


__author__ = 'yuta'
__copyright__ = 'Copyright 2016, Kawai-research'
__license__='GPL'


############################
# 汎用メソッド
############################
@jit
def flatten_with_any_depth(nested_list):
    u"""深さ優先探索の要領で入れ子のリストをフラットにする関数

    Args:
        nested_list:入れ子構造のリスト
    Returns:
        フラットなリスト

    """
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)
        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


def reservoir_sampling(iterable, k):
    u"""iterableなオブジェクトからk個のオブジェクトをランダムに抽出する

    Args:
        iterable:抽出したデータの集合
        k:抽出したい個数
    Returns:
        ランダムに抽出されたk個のアイテム

    """
    it = iter(iterable)
    reservoir = [next(it) for i in xrange(k)]
    n = k
    for item in it:
        n += 1
        r = random.randint(0, n - 1)
        if r < k:
            reservoir[r] = item
    return reservoir


def pp(obj):
    u"""日本語を整形して表示する

    Args:
        obj:整形して表示したいユニコード文字列

    Returns:
        整形されたユニコード文字列

    """
    pp = pprint.PrettyPrinter(indent=4, width=160)
    str = pp.pformat(obj)

    return re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),                                                            16)), str)


def YmdHMS(created_at):
    u"""タイムスタンプをY:M:D:h:m:sに変換

    Args:
        created_at:タイムスタンプ
    Returns:
        YmdHMS時間
    """
    time_utc=time.strptime(created_at,'%a %b %d %H:%M:%S +0000 %Y')
    unix_time=calendar.timegm(time_utc)
    time_local=time.localtime(unix_time)

    return time.strftime("%Y:%m:%d:%H:%M:%S",time_local)


def get_re(data,express,pos):
    u"""データから正規表現にマッチする部分を取得

    Args:
        data:文字列を含むデータ
        express:正規表現
        pos:取得するグループ
    Returns:
        ヒットした部分
    """
    pattern=re.compile(express)
    hit=pattern.search(data)
    if hit:
        return hit.group(pos)
    else:
        return ""


def extract_time(data,type):
    u"""YMDhmsが含まれる文字列から指定したパラメータの値を抽出する

    Args:
        data:時間を抽出したい文字列
        type:抽出するパラメータ
            Y:西暦
            M:月
            D:日
            h:時間
            m:分
            s:秒

    Returns:
        指定されたパラメータの値を文字列で返す

    """
    return get_re(data,r"(?P<Y>\d\d\d\d):(?P<M>\d\d):(?P<D>\d\d):(?P<h>\d\d):(?P<m>\d\d):(?P<s>\d\d)",type)


def write_to_text(data,filename):
    u"""データをファイルに書き込む

    Args:
        data:書き込むデータ
        filename:書き込むファイル
    """
    with codecs.open(filename,"w","utf-8") as f:
        f.write(pp(data))


def save_ndarr(arr,filepath=""):
    u"""numpyオブジェクトを保存する
        Args:
            arr:
            filepath:
    """
    np.savetxt("train.txt",arr,delimiter=",")


############################
# データベース関連
#############################
def load_from_mongo(mongo_db,mongo_db_coll,return_cursor=False,criteria=None,projection=None,limit=0,sort=None,**mongo_conn_kw):
    u"""mongodbから読み込む

    Args:
        mongo_db:db名
        mongo_db_coll:コレクション名
        return_cursor:カーソルとして返すか
        criteria:クエリ条件
        projection:射影条件
        limit:データ数制限
    Returns:
        カーソルかリスト
    """
    client = pymongo.MongoClient(**mongo_conn_kw)
    db=client[mongo_db]
    coll=db[mongo_db_coll]

    if criteria is None:
        criteria={}

    if projection is None:
        if limit==0:
            cursor=coll.find(criteria)
        else:
            cursor=coll.find(criteria).limit(limit)
    else:
        cursor=coll.find(criteria,projection)

    # 大量のデータに対してはカーソルを返す
    if return_cursor:
        return cursor
    else:
        return [item for item in cursor]


def save_to_mongo(data,mongo_db,mongo_db_coll,**mongo_conn_kw):
    u"""mongodbに保存する

    Args:
        data:保存するデータ
        mongo_db:db名
        mongo_db_coll:コレクション名
    """
    # MongoDBサーバーに接続
    client=pymongo.MongoClient(**mongo_conn_kw)
    # 特定のデータベースへの参照を取得する
    db=client[mongo_db]
    # データベースへ内の特定のコレクションへの参照を取得する
    coll=db[mongo_db_coll]
    # バルク挿入を実行してIDを返す
    return coll.save(data)


def delete_mongo(mongo_db,mongo_db_coll,**mongo_conn_kw):
    u"""dbを削除する

    Args:
        mongo_db:db名
        mongo_db_coll:コレクション名
    """
    # メインで使っているコレクションは削除しない
    if mongo_db_coll in ["tweets_of_dropout","tweets_of_dropout?","tweets_of_normal"]:
        print "You can't delete ",mongo_db_coll,"! This is primary collection."
        return
    client = pymongo.MongoClient(**mongo_conn_kw)
    db=client[mongo_db]
    db.drop_collection(mongo_db_coll)


def remove_from_mongo(mongo_db, mongo_db_coll, criteria, **mongo_conn_kw):
    u"""dbから要素を削除する

    Args:
        mongo_db:db名
        mongo_db_coll:コレクション名
        criteria:削除する要素のクエリ

    """
    if mongo_db_coll in ["tweets_of_dropout","tweets_of_dropout?","tweets_of_normal"]:
        print "You can't delete ",mongo_db_coll,"! This is primary collection."
        return
    # MongoDBサーバーに接続
    client=pymongo.MongoClient(**mongo_conn_kw)
    # 特定のデータベースへの参照を取得する
    db=client[mongo_db]
    # データベースへ内の特定のコレクションへの参照を取得する
    coll=db[mongo_db_coll]
    # removeでcriteriaのデータを削除
    coll.remove(criteria)


def create_indexes(mongo_db,mongo_db_coll,indexes,**mongo_conn_kw):
    u"""コレクションへインデックスを貼る
        Args:
            mongo_db:db名
            mongo_db_coll:コレクション名
            indexes:貼るインデックスのリスト
    """
    #MongoDBサーバーに接続
    client=pymongo.MongoClient(**mongo_conn_kw)
    #特定のデータベースへの参照を取得する
    db=client[mongo_db]
    #データベースへ内の特定のコレクションへの参照を取得する
    coll=db[mongo_db_coll]

    if indexes is None:
        return

    #インデックスを貼る
    for index in indexes:
        coll.ensure_index(index)


############################
# 計算関連
#############################
def vec2dense(vec, num_terms):
    u"""sparseフォーマットをdenseに変換
        Args:
            vec:ベクトル(Vectorizer)
            num_terms:トピック数
        Returns:
            dense list
    """
    return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])


def dist_raw(v1,v2):
    u"""特徴ベクトルの類似度を計算する
        Args:
            v1:ベクトル1(Vectorizer)
            v2:ベクトル2(Vectorizer)
        Returns:
            ユークリッド距離
    """
    delta=v1-v2
    return sp.linalg.norm(delta.toarray())


def dist_norm(v1,v2):
    u"""出現回数を正規化したベクトルを返す
        Args:
            v1:ベクトル1(Vectorizer)
            v2:ベクトル2(Vectorizer)
        Returns:
            正規化したベクトル
    """
    v1_normalized=v1/sp.linalg.norm(v1.toarray())
    v2_normalized=v2/sp.linalg.norm(v2.toarray())
    delta=v1_normalized-v2_normalized
    return sp.linalg.norm(delta.toarray())


def similer_document(vectorizer,tweets,newpost):
    u"""文章の類似度を計算する
        Args:
            vectorizer:ベクトル(Vectorizer)
            tweets:文書集
            newpost:類似度を調べる文書
        Returns:
            最も近い文書
    """
    X=vectorizer.fit_transform(tweets)
    num_samples,num_features=X.shape
    new_post=newpost
    new_post_vec=vectorizer.transform([new_post])
    best_dist=sys.maxint
    best_i=None
    for i in range(0,num_samples):
        post=tweets[i]
        if post == new_post:
            continue
        post_vec=X.getrow(i)
        d= dist_norm(post_vec, new_post_vec)
        if d<best_dist:
            best_dist=d
            best_i=i
    return best_i


def tfidf(term,doc,docset):
    u"""TF-IDFを計算する
        Args:
            term:求める文字
            doc:検索ドキュメント
            docset:全ドキュメント
        Returns:
            TF-IDF
    """
    tf=float(doc.count(term))/sum(doc.count(w) for w in set(doc))
    idf=math.log(float(len(docset))/(len([doc for doc in docset
                                          if term in doc])))
    return tf*idf


def comb(n, r):
    u"""組み合わせの数（二項係数）を求める
        Args:
            n:要素数
            r:選択する要素数
        Returns:
            組み合わせの数
    """
    if n == 0 or r == 0: return 1
    return comb(n, r - 1) * (n - r + 1) / r


def binomial(n,p):
    u"""二項分布を求める
        Args:
            n:要素数
            p:確率
        Returns:
            二項分布
    """
    t=0.0
    for k in xrange(n+1):
        b= comb(n, k) * p ** k * (1 - p) ** (n - k)
        t+=b
        print "%d,\t%g,\t%g"%(k,b,t)


def distance(p1, p2):
    u"""N次元配列で二点間のユークリッド距離を求める
        Args:
            p1:配列1
            p2:配列2
        Returns:
            ユークリッド距離
    """
    return np.sum((p1 - p2) ** 2)


############################
# 表示系メソッド
#############################
def show_word_printout():
    u"""普通の学生と中退した学生の特徴度の高い語を表示

    """
    dropouts=load_from_mongo(config.m_dropout.db,config.m_dropout.tahara_coll,True)
    normals=load_from_mongo(config.m_normal.db,config.m_normal.tahara_coll,True)
    import pymongo
    count=0
    print
    print
    print
    print "------------------------"
    print "Dropout Students"
    print "------------------------"
    print
    print
    print
    for d in dropouts.sort("rate",pymongo.DESCENDING):
        print "------------------------"
        print count+1,".",
        print pp(d["_id"])
        print "------------------------"

        count+=1
        if count > 1000:
            break
    count=0
    print
    print
    print
    print "------------------------"
    print "Normal Students"
    print "------------------------"
    print
    print
    print
    for n in normals.sort("rate",pymongo.DESCENDING):
        print "------------------------"
        print count+1,".",
        print pp(n["_id"])
        print "------------------------"
        count+=1
        if count > 1000:
            break


def create_wordcloud(text):
    u"""ワードクラウドを表示

    Args:
        text:分かち書きしたテキスト
    """
    fpath="/Library/Fonts/ヒラギノ角ゴ Pro W3.otf"

    # ストップワードの定義
    stop_words = [ u'てる', u'いる', u'なる', u'れる', u'する', u'ある', u'こと', u'これ', u'さん', u'して', \
             u'くれる', u'やる', u'くださる', u'そう', u'せる', u'した',  u'思う',  \
             u'それ', u'ここ', u'ちゃん', u'くん', u'', u'て',u'に',u'を',u'は',u'の', u'が', u'と', u'た', u'し', u'で', \
             u'ない', u'も', u'な', u'い', u'か', u'ので', u'よう', u''\
                   u"ん",u"もう",u"くる",u"言う",u"行く",u"笑",u"人",u"すぎる",u"私",u"自分",u"回",u"いく",u"何",u"られる"\
                   u"今日",u"なん",u"バイト",u"勉強",u"ー"]

    wordcloud = WordCloud(background_color="white",font_path=fpath, width=900, height=500, \
                          stopwords=set(stop_words)).generate(text)
    plt.figure(figsize=(15,12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def dprint(text):
    u"""デバッグモードの時だけプリントする
        Args:
            text:
    """
    if config.DEBUG is True:
        print pp(text)
    else :
        pass


############################
# 言語処理系メソッド
#############################
def get_stopwords():
    u"""ストップワードを取得する
        Returns:
            ストップワードのセット
    """
    res = set([ u"!", u"\"", u"#", u"$", u"%", u"&", u"'", u"(", u")", u"*", u"+", u",", u"-", u".", u"/", u"0", u"1", u"10", u"11", u"12", u"13", u"14", u"15", u"16", u"17", u"18", u"19", u"2", u"20", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u":", u";", u"<", u"=", u">", u"?", u"@", u"A", u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"I", u"J", u"K", u"L", u"M", u"N", u"O", u"P", u"Q", u"S", u"T", u"U", u"V", u"W", u"X", u"Y", u"Z", u"[", u"\\", u"]", u"^", u"_", u"`", u"a", u"b", u"c", u"d", u"e", u"f", u"g", u"h", u"i", u"j", u"k", u"l", u"m", u"n", u"o", u"p", u"q", u"s", u"t", u"u", u"v", u"w", u"x", u"y", u"z", u"{", u"|", u"}", u"~", u"ω", u"д", u"‐", u"‘", u"’", u"”", u"ぁ", u"あ", u"あそこ", u"あたり", u"あちら", u"あっち", u"あと", u"あな", u"あなた", u"ある", u"あれ", u"ぃ", u"い", u"いい", u"いう", u"いえる", u"いか", u"いく", u"いくつ", u"いつ", u"いま", u"いる", u"いろいろ", u"ぅ", u"う", u"うち", u"ぇ", u"え", u"ぉ", u"お", u"おおまか", u"おこなう", u"おこる", u"おまえ", u"おもう", u"おれ", u"お前", u"か", u"かく", u"かたちの", u"かやの", u"から", u"かれる", u"かんがえる", u"が", u"がい", u"がら", u"き", u"きた", u"きみ", u"きゃ", u"きゅ", u"きょ", u"ぎ", u"く", u"くださる", u"くらべる", u"くる", u"くれる", u"くん", u"ぐ", u"け", u"げ", u"こ", u"ここ", u"こせ", u"こちら", u"こっち", u"こと", u"これ", u"これら", u"ご", u"ごと", u"ごろ", u"さ", u"さまざま", u"さらい", u"さん", u"ざ", u"し", u"しかた", u"した", u"して", u"しまう", u"しゃ", u"しゅ", u"しょ", u"しよう", u"しれる", u"じ", u"す", u"すか", u"すぎ", u"すぎる", u"すね", u"する", u"ず", u"ずつ", u"せ", u"せる", u"ぜ", u"そ", u"そう", u"そこ", u"そちら", u"そっち", u"そで", u"そのもの", u"それ", u"それぞれ", u"それなり", u"ぞ", u"た", u"たくさん", u"たち", u"たび", u"ため", u"だ", u"ち", u"ちがう", u"ちゃ", u"ちゃう", u"ちゃん", u"ちゅ", u"ちょ", u"ぢ", u"っ", u"つ", u"つぎ", u"つく", u"つくる", u"つける", u"つまり", u"づ", u"て", u"てる", u"てるん", u"てん", u"で", u"できる", u"でる", u"と", u"とおり", u"とき", u"とこ", u"ところ", u"とる", u"ど", u"どこ", u"どこか", u"どちら", u"どれ", u"な", u"ない", u"なか", u"なかば", u"ながい", u"なす", u"など", u"なに", u"なる", u"なん", u"に", u"にゃ", u"にゅ", u"にょ", u"ぬ", u"ね", u"の", u"ので", u"は", u"はいる", u"はじめ", u"はず", u"はるか", u"ば", u"ぱ", u"ひ", u"ひと", u"ひとつ", u"ひゃ", u"ひゅ", u"ひょ", u"び", u"ぴ", u"ふ", u"ふく", u"ぶ", u"ぶり", u"ぷ", u"へ", u"へん", u"べ", u"べつ", u"べん", u"ぺ", u"ほ", u"ほう", u"ほか", u"ぼ", u"ぼく", u"ぽ", u"ま", u"まさ", u"まし", u"まとも", u"まま", u"み", u"みたい", u"みつ", u"みなさん", u"みゃ", u"みゅ", u"みょ", u"みる", u"みんな", u"む", u"め", u"も", u"もちいる", u"もつ", u"もと", u"もの", u"もらう", u"もん", u"ゃ", u"や", u"やつ", u"やる", u"ゅ", u"ゆ", u"ょ", u"よ", u"よい", u"よう", u"よそ", u"よぶ", u"よる", u"ら", u"られる", u"り", u"りゃ", u"りゅ", u"りょ", u"る", u"れ", u"れる", u"ろ", u"ゎ", u"わ", u"わかる", u"わが国", u"わけ", u"わたし", u"ゐ", u"ゑ", u"を", u"ん", u"ゔ", u"ァ", u"ア", u"アソコ", u"アタリ", u"アチラ", u"アッチ", u"アト", u"アナ", u"アナタ", u"アル", u"アレ", u"ィ", u"イ", u"イウ", u"イカ", u"イクツ", u"イツ", u"イマ", u"イル", u"イロイロ", u"ゥ", u"ウ", u"ウチ", u"ェ", u"エ", u"ォ", u"オ", u"オオマカ", u"オマエ", u"オレ", u"カ", u"カク", u"カタチノ", u"カヤノ", u"カラ", u"カ所", u"カ月", u"ガ", u"ガイ", u"ガラ", u"キ", u"キタ", u"ギ", u"ク", u"クダサル", u"クル", u"クレル", u"クン", u"グ", u"ケ", u"ゲ", u"コ", u"ココ", u"コセ", u"コチラ", u"コッチ", u"コト", u"コレ", u"コレラ", u"ゴ", u"ゴト", u"ゴロ", u"サ", u"サマザマ", u"サライ", u"サン", u"ザ", u"シ", u"シカタ", u"シタ", u"シテ", u"シマウ", u"シヨウ", u"ジ", u"ス", u"スカ", u"スギ", u"スギル", u"スネ", u"スル", u"ズ", u"ズツ", u"セ", u"セル", u"ゼ", u"ソ", u"ソウ", u"ソコ", u"ソチラ", u"ソッチ", u"ソデ", u"ソレ", u"ソレゾレ", u"ソレナリ", u"ゾ", u"タ", u"タクサン", u"タチ", u"タビ", u"タメ", u"ダ", u"チ", u"チャ", u"チャウ", u"チャン", u"ヂ", u"ッ", u"ツ", u"ツク", u"ヅ", u"テ", u"テル", u"テルン", u"テン", u"デ", u"ト", u"トオリ", u"トキ", u"トコロ", u"ド", u"ドコ", u"ドコカ", u"ドチラ", u"ドレ", u"ナ", u"ナカ", u"ナカバ", u"ナド", u"ナニ", u"ナル", u"ナン", u"ニ", u"ヌ", u"ネ", u"ノ", u"ノデ", u"ハ", u"ハジメ", u"ハズ", u"ハルカ", u"バ", u"パ", u"ヒ", u"ヒト", u"ヒトツ", u"ビ", u"ピ", u"フ", u"フク", u"ブ", u"ブリ", u"プ", u"ヘ", u"ヘン", u"ベ", u"ベツ", u"ベン", u"ペ", u"ホ", u"ホウ", u"ホカ", u"ボ", u"ボク", u"ポ", u"マ", u"マサ", u"マシ", u"マトモ", u"ママ", u"ミ", u"ミタイ", u"ミツ", u"ミナサン", u"ミル", u"ミンナ", u"ム", u"メ", u"モ", u"モト", u"モノ", u"モン", u"ャ", u"ヤ", u"ヤツ", u"ヤル", u"ュ", u"ユ", u"ョ", u"ヨ", u"ヨウ", u"ヨソ", u"ラ", u"ラレル", u"リ", u"ル", u"レ", u"レル", u"ロ", u"ワ", u"ワケ", u"ワタシ", u"ヲ", u"ン", u"ヴ", u"ヵ所", u"ヶ所", u"ヶ月", u"ー", u"一", u"一つ", u"七", u"万", u"三", u"上", u"上記", u"下", u"下記", u"中", u"九", u"事", u"二", u"五", u"人", u"人々", u"人びと", u"今", u"今回", u"今日", u"他", u"以上", u"以下", u"以前", u"以後", u"以降", u"何", u"何人", u"作る", u"使う", u"例", u"俺", u"個", u"僕", u"億", u"兆", u"先", u"入る", u"入れる", u"全部", u"八", u"六", u"円", u"冬", u"出す", u"出る", u"出来る", u"分", u"別", u"前", u"前回", u"化", u"匹", u"区", u"十", u"千", u"半ば", u"右", u"各", u"同じ", u"名", u"名前", u"君", u"品", u"四", u"回", u"国", u"土", u"場合", u"夏", u"多い", u"大きい", u"字", u"家", u"小さい", u"少ない", u"左", u"市", u"席", u"年", u"府", u"度", u"強い", u"形", u"後", u"得る", u"思う", u"悪い", u"感じ", u"所", u"手", u"手段", u"持つ", u"方", u"日", u"明日", u"春", u"時", u"時点", u"時間", u"書く", u"月", u"木", u"未満", u"本当", u"村", u"束", u"来る", u"様々", u"次", u"歳", u"毎", u"毎日", u"気", u"水", u"火", u"点", u"用", u"用いる", u"町", u"異なる", u"百", u"的", u"目", u"県", u"知る", u"確か", u"示す", u"私", u"秋", u"秒", u"笑", u"第", u"等", u"箇所", u"箇月", u"系", u"結局", u"考える", u"者", u"聞く", u"自体", u"自分", u"良い", u"行う", u"行く", u"行なう", u"見える", u"見る", u"言う", u"話", u"誰", u"起こる", u"身", u"述べる", u"週", u"過ぎる", u"道", u"達", u"違う", u"都", u"金", u"長い", u"間", u"際", u"類", u"高い", u"！", u"＃", u"＄", u"％", u"＆", u"（", u"）", u"＊", u"＋", u"，", u"－", u"．", u"／", u"０", u"１", u"１０", u"１１", u"１２", u"１３", u"１４", u"１５", u"１６", u"１７", u"１８", u"１９", u"２", u"２０", u"３", u"４", u"５", u"６", u"７", u"８", u"９", u"：", u"；", u"＜", u"＝", u"＞", u"？", u"＠", u"Ａ", u"Ｂ", u"Ｃ", u"Ｄ", u"Ｅ", u"Ｆ", u"Ｇ", u"Ｈ", u"Ｉ", u"Ｊ", u"Ｋ", u"Ｌ", u"Ｍ", u"Ｎ", u"Ｏ", u"Ｐ", u"Ｑ", u"Ｓ", u"Ｔ", u"Ｕ", u"Ｖ", u"Ｗ", u"Ｘ", u"Ｙ", u"Ｚ", u"［", u"］", u"＾", u"＿", u"ａ", u"ｂ", u"ｃ", u"ｄ", u"ｅ", u"ｆ", u"ｇ", u"ｈ", u"ｉ", u"ｊ", u"ｋ", u"ｌ", u"ｍ", u"ｎ", u"ｏ", u"ｐ", u"ｑ", u"ｓ", u"ｔ", u"ｕ", u"ｖ", u"ｗ", u"ｘ", u"ｙ", u"ｚ", u"｛", u"｜", u"｝", u"～", u"ｦ", u"ｧ", u"ｨ", u"ｩ", u"ｪ", u"ｫ", u"ｬ", u"ｭ", u"ｮ", u"ｯ", u"ｱ", u"ｱｯﾁ", u"ｱｿｺ", u"ｱﾀﾘ", u"ｱﾁﾗ", u"ｱﾄ", u"ｱﾅ", u"ｱﾅﾀ", u"ｱﾙ", u"ｱﾚ", u"ｲ", u"ｲｳ", u"ｲｶ", u"ｲｸﾂ", u"ｲﾂ", u"ｲﾏ", u"ｲﾙ", u"ｲﾛｲﾛ", u"ｳ", u"ｳﾁ", u"ｳﾞ", u"ｴ", u"ｵ", u"ｵｵﾏｶ", u"ｵﾏｴ", u"ｵﾚ", u"ｶ", u"ｶｸ", u"ｶﾀﾁﾉ", u"ｶﾔﾉ", u"ｶﾗ", u"ｶﾞ", u"ｶﾞｲ", u"ｶﾞﾗ", u"ｷ", u"ｷﾀ", u"ｷﾞ", u"ｸ", u"ｸﾀﾞｻﾙ", u"ｸﾙ", u"ｸﾚﾙ", u"ｸﾝ", u"ｸﾞ", u"ｹ", u"ｹﾞ", u"ｺ", u"ｺｯﾁ", u"ｺｺ", u"ｺｾ", u"ｺﾁﾗ", u"ｺﾄ", u"ｺﾚ", u"ｺﾚﾗ", u"ｺﾞ", u"ｺﾞﾄ", u"ｺﾞﾛ", u"ｻ", u"ｻﾏｻﾞﾏ", u"ｻﾗｲ", u"ｻﾝ", u"ｻﾞ", u"ｼ", u"ｼｶﾀ", u"ｼﾀ", u"ｼﾃ", u"ｼﾏｳ", u"ｼﾖｳ", u"ｼﾞ", u"ｽ", u"ｽｶ", u"ｽｷﾞ", u"ｽｷﾞﾙ", u"ｽﾈ", u"ｽﾙ", u"ｽﾞ", u"ｽﾞﾂ", u"ｾ", u"ｾﾙ", u"ｾﾞ", u"ｿ", u"ｿｯﾁ", u"ｿｳ", u"ｿｺ", u"ｿﾁﾗ", u"ｿﾃﾞ", u"ｿﾚ", u"ｿﾚｿﾞﾚ", u"ｿﾚﾅﾘ", u"ｿﾞ", u"ﾀ", u"ﾀｸｻﾝ", u"ﾀﾁ", u"ﾀﾋﾞ", u"ﾀﾒ", u"ﾀﾞ", u"ﾁ", u"ﾁｬ", u"ﾁｬｳ", u"ﾁｬﾝ", u"ﾁﾞ", u"ﾂ", u"ﾂｸ", u"ﾂﾞ", u"ﾃ", u"ﾃﾙ", u"ﾃﾙﾝ", u"ﾃﾝ", u"ﾃﾞ", u"ﾄ", u"ﾄｵﾘ", u"ﾄｷ", u"ﾄｺﾛ", u"ﾄﾞ", u"ﾄﾞｺ", u"ﾄﾞｺｶ", u"ﾄﾞﾁﾗ", u"ﾄﾞﾚ", u"ﾅ", u"ﾅｶ", u"ﾅｶﾊﾞ", u"ﾅﾄﾞ", u"ﾅﾆ", u"ﾅﾙ", u"ﾅﾝ", u"ﾆ", u"ﾇ", u"ﾈ", u"ﾉ", u"ﾉﾃﾞ", u"ﾊ", u"ﾊｼﾞﾒ", u"ﾊｽﾞ", u"ﾊﾙｶ", u"ﾊﾞ", u"ﾊﾟ", u"ﾋ", u"ﾋﾄ", u"ﾋﾄﾂ", u"ﾋﾞ", u"ﾋﾟ", u"ﾌ", u"ﾌｸ", u"ﾌﾞ", u"ﾌﾞﾘ", u"ﾌﾟ", u"ﾍ", u"ﾍﾝ", u"ﾍﾞ", u"ﾍﾞﾂ", u"ﾍﾞﾝ", u"ﾍﾟ", u"ﾎ", u"ﾎｳ", u"ﾎｶ", u"ﾎﾞ", u"ﾎﾞｸ", u"ﾎﾟ", u"ﾏ", u"ﾏｻ", u"ﾏｼ", u"ﾏﾄﾓ", u"ﾏﾏ", u"ﾐ", u"ﾐﾀｲ", u"ﾐﾂ", u"ﾐﾅｻﾝ", u"ﾐﾙ", u"ﾐﾝﾅ", u"ﾑ", u"ﾒ", u"ﾓ", u"ﾓﾄ", u"ﾓﾉ", u"ﾓﾝ", u"ﾔ", u"ﾔﾂ", u"ﾔﾙ", u"ﾕ", u"ﾖ", u"ﾖｳ", u"ﾖｿ", u"ﾗ", u"ﾗﾚﾙ", u"ﾘ", u"ﾙ", u"ﾚ", u"ﾚﾙ", u"ﾛ", u"ﾜ", u"ﾜｹ", u"ﾜﾀｼ", u"ﾝ", u"￥", ])

    return res


def get_word_point(word):
    u"""関数の説明
        Args:
            hoge:引数の説明
        Returns:
            foo:戻り値の説明
    """
    #要素数80
    res={u"死ぬ":1,u"無理":1,u"痛い":1,u"やめる":2,u"怖い":1,u"泣く":1,u"辞める":3,u"嫌い":1,u"だるい":1,u"しんどい":1,u"つらい":1,
         u"殺す":1,u"怒り":1,u"つかれる":1,u"怒る":1,u"疲れる":1,u"落ちる":1,u"落とす":2,u"ねむい":1,u"がる":1,
         u"めんどくさい":1,u"退学":3,u"おかしい":1,u"きつい":1,u"死ねる":1,u"嫌":1,u"辛い":1,u"休む":1,u"遅刻":1,
         u"怒り怒り":1,u"悩む":1,u"届":1,u"いたい":1,u"こわい":1,u"忙しい":1,u"自殺":2,u"負ける":1,u"無駄":1,
         u"諦める":2,u"耐える":1,u"めんどい":1,u"邪魔":1,u"クズ":1,u"退学届":3,u"だめ":1,u"行きたくない":2,
         u"リスカ":3,u"ひどい":1,u"寂しい":1,u"頭痛い":1,u"朝早い":1,u"学校辞める":1,u"寝る寝る":1,u"イラ":1,
         u"いらいら":1,u"妊娠":1,u"病む":2,u"腹立つ":1,u"寝不足":1,u"泣き":1,u"怒り怒り怒り":1,u"地獄":2,u"恐ろしい":1,
         u"迷う":1,u"死":1,u"後悔":1,u"悩み":1,u"学校やめる":3,u"怒り怒り怒り怒り":2,u"産まれる":1,u"無視":1,u"やめるほしい":2,
         u"大学辞める":3,u"迷惑かける":1,u"落胆の表情":1,u"苦手":1,u"無理無理":1,u"泣き顔泣き顔泣き顔":1,u"ストレス":1,
         u"ムリ":1,}
    if word in res:
        ret = res[word]
    else:
        ret=0
    return ret


def load_stopwords_old():
    u"""前に使ってたストップワード
    """
    with codecs.open("read/stopwords.txt","r","utf-8") as f:
            lines=f.readlines()
    res=[line.rstrip("\n") for line in lines]
    return res
