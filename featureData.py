#encoding=utf-8

import cgi

header = 'Content-Type: text/plain; charset=utf-8\r\n'

rtn = ''

VALID_FEAT_NUM = [344, 786, 967, 1282, 1627]
VECTORIZE_PATH = 'features/'

def process():
    form = cgi.FieldStorage()
    f = 967
    print header
    
    try:
        if form.has_key('f'):
            tmpf = int(form['f'].value)
            if VALID_FEAT_NUM.count(tmpf) > 0:
                f = tmpf
        fname = VECTORIZE_PATH + 'featsByDFandIG-' + str(f) + '.txt'
        n = 'featsByDFandIG-' + str(f) + '.txt'
        print '### Filename : %s , 特征维数: %d ###' % (n, f)##;exit()
        print '### 词项:信息增益(IG) ' + '#'*35
        fh = open(fname, 'r')
        print fh.read()
        fh.close()
    except Exception, msg:
        print msg
    
if __name__=="__main__":
    process()
