#encoding=utf-8

import cgi
import sys
import os
import json
import re
import urllib2
from urlparse import *
from bs4 import BeautifulSoup
import StringIO
import gzip

import classify

REQUESTHeader = {
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding':'gzip',##可能会压缩，需要处理
    'Accept-Language':'zh-CN,zh;q=0.8',
    'Cache-Control':'max-age=0',
    'Connection':'keep-alive',
    'Host':'',
    'Referer':'http://www.baidu.com',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36'
    }
ARTICLE_IDS = {
    'sina.com' : 'artibody',
    'qq.com' : 'Cnt-Main-Article-QQ',
    'sohu.com' : 'contentText',
    '163.com': 'endText'
    }
CLASS_LABEL = {
    'C000008':[1, u'财经'],
    'C000010':[2, 'IT'],
    'C000013':[3, u'健康'],
    'C000014':[5, u'体育'],
    'C000016':[4, u'旅游'],
    'C000020':[6, u'教育'],
    'C000022':[7, u'招聘'],
    'C000023':[8, u'文化'],
    'C000024':[9, u'军事']
    }
IDF = 'idf.txt'
FEAT = 'features/featsByDFandIG-967.txt'
CLASSIFY = 'vectorize/vectorize-tfidf-967.csv'


header = 'Content-Type: text/json; charset=utf-8\r\n'
rtn = {'status': 1, 'info': 'OK'}


def getArticleId(url):
    ''' The specific article content div id in total 4 websites'''
    urlParts = urlparse(url)
    host = urlParts[1] ## netloc
    aid = ''
    for d in ARTICLE_IDS:
        if host.count(d) > 0:
            aid = ARTICLE_IDS[d]
            break
    return aid

def getArticle(url):
    '''Get the article content by the only one url'''
    urls = urlparse(url, '')
    if urlunparse(urls) != url:
        raise Exception('Invalid url')
    aHeaders = REQUESTHeader
    aHeaders['Host'] = urls[1]
    
    req = urllib2.Request(url)
    for h in aHeaders:
        req.add_header(h, aHeaders[h])
    response = urllib2.urlopen(req)
    con = response.read()
    isGzip = response.headers.get('Content-Encoding')
    
    ##处理GZIP压缩问题
    if isGzip.lower() == 'gzip':
        compressStream = StringIO.StringIO(con)
        gzipper = gzip.GzipFile(fileobj = compressStream)
        con = gzipper.read()
    
    uniStr =  con.decode('gb18030', 'ignore')  ##中文字符串解码为unicode字符串
    
    ## Parse the html content
    soup = BeautifulSoup(uniStr, from_encoding='utf-8')
    aid = getArticleId(url); ## Find the article content div-id
    a = soup.find(id = aid)  ## a = a.text会产生去除标签
    s = str(a)               ## Tag对象转换为字符串（utf-8）
    s = re.sub(ur'[\r\n\a\b\f\v\t ]+?', '', s)
    s = re.sub(ur'<\s*style[\s\S]+?<\s*/style\s*>', '', s)
    s = re.sub(ur'<\s*script[\s\S]+?<\s*/script\s*>', '', s)
    s = re.sub(ur'<!--[\s\S]+?-->', '', s)
    s = re.sub(ur'<br\s*?/?>', '', s)
    s = re.sub(ur'<\s*\[[^\]]+?\]\s*>[^>]+?<\s*\[[^\]]+?\]\s*>', '', s)
    s = re.sub(ur'&[\w]+;', '', s)
    s = re.sub(ur'<[^>]+?>', '', s)
    
    s = s.decode('utf-8', 'ignore')  ##将utf-8字符串转换为unicode字符串
    ##print s.encode('gb18030')
    return s

def commonClassify(article):
    result = {}
    ds, labels = classify.loadDataSet(CLASSIFY, 5400, 967)
    x = classify.vectorArticleByTFIDF(article, FEAT, IDF)
    c1 = int(classify.classifyBykNN(ds, labels, 10, x))
    c2 = int(classify.classifyByRocchio(ds, labels, x))
    c3 = int(classify.classifyByNBC(x)[0])
    c4 = int(classify.classifyBySVM(x)[0])
    c5 = int(classify.classifyByANN(x)[0])
    result['kNN'] = [];    result['kNN'].append(c1)
    result['Rocchio'] = [];result['Rocchio'].append(c2)
    result['NBC'] = [];    result['NBC'].append(c3)
    result['SVM'] = [];    result['SVM'].append(c4)
    result['ANN'] = [];    result['ANN'].append(c5)
    for c in CLASS_LABEL:
        cid = CLASS_LABEL[c][0]
        cname = CLASS_LABEL[c][1]
        for r in result:
            if result[r][0] == cid:
                result[r].append(c)
                result[r].append(cname)
    return result

def process():
    ''' Cgi process '''
    form = cgi.FieldStorage()
    try:
        if not form.has_key('url'):
            raise Exception('No url field')
        url = form['url'].value
        url = url.strip()
        a = getArticle(url)
        r = commonClassify(a);
        
        rtn['info'] = {}
        rtn['info'][url] = r
        demo = {##'kNN':[1, 'C000016', '财经'],
                ##'Rocchio':[2,'C000016','财经'],
                ##'NBC':[2,'C000016','财经'],
                ##'SVM':[2,'C000016','财经'],
                ##'ANN':[2,'C000016','财经']
                }
        ##rtn['info'][url].update(demo)
    except Exception, msg:
        rtn['status'] = 0
        rtn['info'] = str(msg)
    print header
    print json.dumps(rtn)

if __name__=="__main__":
    process()
