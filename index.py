#encoding=utf-8

import cgi

header = 'Content-Type: text/html; charset=utf-8\r\n'

def process():
    html = open('html/index.html', 'r').read()
    print header
    print html

if __name__=="__main__":
    process()
