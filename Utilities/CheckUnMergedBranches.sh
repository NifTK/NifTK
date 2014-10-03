#!/bin/sh

git fetch origin

git stash

current_branch=$(git rev-parse --abbrev-ref HEAD)

for b in master sls thifu
do
  git branch -a --merged $b > /tmp/$b.txt
done

for b in `git branch -r | grep "^\ *origin/" | sed s,origin/,, | grep -v HEAD | grep -v ">" | grep -v master | grep -v sls | grep -v thifu
do
  FOUND_IT=0
  for c in master sls thifu
  do
    if grep -q $b /tmp/$c.txt
    then
      FOUND_IT=1
    fi
  done

  if [ $FOUND_IT -eq 1 ]; then
    echo $b >> branches-merged.txt
  else
    echo $b >> branches-not-merged.txt
    git log --pretty=format:"%h - %an, %ar : %s" $b | head -3 >> branches-not-merged.txt
  fi

done

for b in master sls thifu
do
  rm /tmp/$b.txt
done

git checkout $current_branch

git stash pop
 
