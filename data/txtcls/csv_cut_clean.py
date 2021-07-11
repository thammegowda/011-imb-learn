#!/usr/bin/env python
import csv, sys, html
if len(sys.argv) != 2:
    print(sys.argv)
    raise Exception(f'usage: {sys.argv[0]} <column#> ')
col = int(sys.argv[1])
# escapechar="\\\\"
for rec in csv.reader(sys.stdin, quotechar='"'):
    txt = rec[col].strip()
    txt = txt.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
    #txt = html.unescape(txt)
    txt = " ".join(txt.split())  # remove unwanted white spaces
    print(txt)
    
