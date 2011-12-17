#!/bin/bash

set -x 

# Just to capture the std output. ADD_TEST doesn't explicitly do it. 
$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} "" > ${18}

cat ${18}
bsi=`cat ${18} | awk -F, '{printf $55}'`
echo ${bsi} > ${18}

