#!/usr/bin/env python 
import os
import sys
import re
#1 function to clean single ipynb

def post_process_ipynb(input_file, output_file=None):
    """
    Reads an ipynb file or takes an ipynb string.

    Removes certain elements produced by doconce that prevent proper processing by ipython

    """
    if os.path.isfile(input_file):
        with open(input_file) as ipynb:
            ipynb = ipynb.read()
    elif isinstance(input_file, str):
        ipynb = input_file
    else:
        raise TypeError("input_file:{} is neither a valid path nor a string")

    clean_ipynb = parse_ipynb(ipynb)
    
    if output_file is None:
        output_file = input_file

    with open(output_file, 'w') as f:
        f.write(clean_ipynb)

def parse_ipynb(ipynb):
    """
    Removes matching items from the string
    """
    ipynb = re.sub("\\\\.*?label\{.*?\}","",ipynb) 
    # \\\\ matches only one \
    ipynb = re.sub(r"%matplotlib inline","",ipynb)
    return ipynb


#2 iterate through all ipynb's and clean

if __name__ == "__main__":
    if len(sys.argv[1:])>=1:
        infile =  sys.argv[1]
    else:
        print("post_do_clean <infile> <outfile>")
        exit()

    if len(sys.argv[1:])>=2:
        outfile = sys.argv[2]
    else:
        outfile = None

    if len(sys.argv[1:])>2:
        print("Too many arguments")
        exit()

    post_process_ipynb(infile, output_file=outfile)

