#encoding=utf-8

import cgi
import sys
import os
import json

import urlClassify

header = 'Content-Type: text/plain; charset=utf-8\r\n'
rtn = {'status' : 1, 'info' : 'OK'}

def process():
    ''' Cgi process '''
    print header
    form = cgi.FieldStorage()
    try:
        if not form.has_key('urlFile'):
            raise Exception('No url file selected')
        urlFile = form['urlFile']
        fp = urlFile.file
        urls = fp.readlines()
        rtn['info'] = {}
        for url in urls:
            url = url.strip()
            a = urlClassify.getArticle(url)
            r = urlClassify.commonClassify(a)
            rtn['info'][url] = r
        fp.close()
    except Exception, msg:
        rtn['status'] = 0
        rtn['info'] = str(msg)
    print json.dumps(rtn)

if __name__=="__main__":
    process()
