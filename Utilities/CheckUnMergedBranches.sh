#!/bin/sh

git fetch origin

rm branches-merged.txt branches-not-merged.txt 2> /dev/null

git stash save --include-untracked | grep -qv "No local changes to save"
stashed_changes=$?

current_branch=$(git rev-parse --abbrev-ref HEAD)

for main_branch in master sls thifu
do
  git branch -a --merged origin/$main_branch > /tmp/$main_branch.txt
done

for feature_branch in `git branch -r | grep "^\ *origin/" | sed s,origin/,, | grep -v HEAD | grep -v ">" | grep -v master | grep -v sls | grep -v thifu`
do
  FOUND_IT=0
  for main_branch in master sls thifu
  do
    if grep -q $feature_branch /tmp/$main_branch.txt
    then
      FOUND_IT=1
    fi
  done

  if [ $FOUND_IT -eq 1 ]; then
    echo $feature_branch >> branches-merged.txt
  else
    echo $feature_branch >> branches-not-merged.txt
    git log --pretty=format:"%h - %an, %ar : %s" "origin/$feature_branch" | head -3 >> branches-not-merged.txt
  fi

done

for main_branch in master sls thifu
do
  rm /tmp/$main_branch.txt
done

git checkout $current_branch

if [ $stashed_changes -eq 0 ]
then
  git stash pop
fi
