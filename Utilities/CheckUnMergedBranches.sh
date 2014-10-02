#!/bin/sh

for b in dev sls midas thifu EpiNav
do
  git checkout $b > /dev/null 2>&1
  git branch -a --merged > /tmp/$b.txt
done

for b in `git branch -r | sed s,origin/,, | grep -v HEAD | grep -v ">" `
do
  FOUND_IT=0
  for c in dev sls midas thifu EpiNav
  do
    WC=`cat /tmp/$c.txt | grep $b`
    if [ "x${WC}x" != "xx" ]; then 
      FOUND_IT=1
    fi
  done

  if [ $FOUND_IT -eq 1 ]; then
    echo $b >> branches-merged.txt
  else
    echo $b >> branches-not-merged.txt
    git checkout $b
    git log --pretty=format:"%h - %an, %ar : %s" | head -3 >> branches-not-merged.txt
  fi

done

for b in dev sls midas thifu EpiNav
do
  rm /tmp/$b.txt
done

git checkout dev
 
