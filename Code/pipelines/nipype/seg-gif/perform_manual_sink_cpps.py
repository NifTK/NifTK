#! /usr/bin/env python

import argparse
import os
import shutil
import time

parser = argparse.ArgumentParser(description='Manually Sinking the CPP files')
parser.add_argument('-ref', '--reference',
                    dest='reference',
                    metavar='reference',
                    help='reference files list',
                    required=True)
parser.add_argument('-flo', '--floating',
                    dest='floating',
                    metavar='floating',
                    help='floating files list',
                    required=True)
parser.add_argument('-cpp', '--cpp',
                    dest='cpp',
                    metavar='cpp',
                    help='cpp files list',
                    required=True)
parser.add_argument('-invcpp', '--invcpp',
                    dest='invcpp',
                    metavar='invcpp',
                    help='invcpp files list',
                    required=True)

parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the template library is stored',
                    required=True)

args = parser.parse_args()


'''
Read the text files:
'''

refs    = [line.rstrip() for line in open(args.reference)]
flos    = [line.rstrip() for line in open(args.floating)]
cpps    = [line.rstrip() for line in open(args.cpp)]
invcpps = [line.rstrip() for line in open(args.invcpp)]

print 'Lines in ref, flo, cpp, invcpp are: ', len(refs), len(flos), len(cpps), len(invcpps)

print 'Start copying files in 5 seconds...'

time.sleep(5)

N = len(refs)

id_matrix_content = '1 0 0 0 \n' + \
                    '0 1 0 0 \n' + \
                    '0 0 1 0 \n' + \
                    '0 0 0 1'

for i in range(refs):
    
    ref = refs[i]
    flo = flos[i]
    cpp = cpps[i]
    invcpp = invcpps[i]
    
    ref_b = os.path.basename(ref)[:-28]
    flo_b = os.path.basename(flo)[:-28]
    
    print i, ref_b, flo_b
    
    cpp_target = os.path.join(args.output, ref_b, flo_b + '.nii.gz')
    invcpp_target = os.path.join(args.output, flo_b, ref_b + '.nii.gz')

    shutil.copyfile(cpp, cpp_target)

    shutil.copyfile(invcpp, invcpp_target)
    

