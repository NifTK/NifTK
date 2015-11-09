#!/bin/bash

# This script checks if an external project (EP) build directory is still needed by
# one of the current branches. If yes, it prints the name of the first branch that
# requires the that version of the EP. Otherwise, it deletes the EP directory.
#
# Usage:
#
# niftk-ep-clean-up.sh <source directory> <external project directory>

if [ $# -ne 2 ]
then
  echo "Usage:"
  echo
  echo "    niftk-ep-clean-up.sh <source directory> <external project directory>"
  echo
  exit 1
fi

SRC_DIR="$1"
EP_DIR="$2"

cd "$SRC_DIR"
branches=`git branch -r | grep -v " -> "`

cd "$EP_DIR"
EPs=$(ls -1d "*/*")

for EP in $EPs
do

  ep_name=${EP%%/*}
  ep_version=${EP#*/}
  echo -n "$ep_name/$ep_version: "

  cd "$SRC_DIR"

  ep_needed_by=""
  for branch in $branches
  do
    ep_version_on_branch=`git log -n 1 --pretty=format:%h $branch -- CMake/CMakeExternals/$ep_name.cmake | cut -c -5`
    if [ "$ep_version" == "$ep_version_on_branch" ]
    then
      ep_needed_by=$branch
      break
    fi
  done

  cd "$EP_DIR"

  if [ "$ep_needed_by" == "" ]
  then
    echo "not needed any more. Deleting it."
    rm -rf "$EP"
  else
    echo "still needed by $ep_needed_by."
  fi

  # Delete the directory if it is empty.
  if ! [ "$(ls -A $ep_name)" ]
  then
    rmdir $ep_name
  fi

done
