#!/bin/bash

# Downloads external project dependencies of NifTK and creates a
# tarball and MD5 checksum.

# Note:
# We have to use an old version of subversion so that the external projects
# can be checked out on each client.
# This script is supposed to be used on jet.cs.ucl.ac.uk where there is
# Subversion 1.4.0 in /opt/subversion.

export PATH=/opt/subversion/bin:$PATH

function print_usage() {
  echo "
Usage:

    download-tarball.sh [-s|--sources] [-r|--repository] <project> <version>

Options:

    -s, --sources         Put the source files in the tarball.

    -r, --repository      Put the files of the versioning system in the tarball.

The options can be combined but at least one of them must be given. The suggested
options are -r for the git and -sr for the subversion projects.

Supported projects:

    MITK
    commontk/CTK
    NifTK/CTK
    OpenIGTLink
    SlicerExecutionModel
    qRestAPI
    NiftyRec
    NiftyReg
    NiftySeg
    NiftySim
    NiftyLink
    NifTKData
    IGSTK
"
  exit 1
}

if [ $# -eq 0 ]
then
  print_usage
fi

while true
do
  if [[ $1 = -s || $1 = --sources ]]
  then
    keep_sources=true
    shift
  elif [[ $1 = -r || $1 = --repository ]]
  then
    keep_repository=true
    shift
  elif [[ $1 = -sr || $1 = -rs ]]
  then
    keep_sources=true
    keep_repository=true
    shift
  else
    break
  fi
done

if [ $# -ne 2 ]
then
  print_usage
fi

if [[ ! $keep_sources && ! $keep_repository ]]
then
  print_usage
fi

project=$1
version=$2

function download_from_github() {
  organisation=$1
  project=$2
  version=$3
  if [ -z $4 ]
  then
    directory=$organisation-$project-$version
  else
    directory=$4
  fi
  tarball=$directory.tar.gz
  if [[ $keep_sources && ! $keep_repository ]]
  then
    rm $tarball 2> /dev/null
    wget -O $tarball http://github.com/$organisation/$project/tarball/$version
  elif [[ $keep_repository && ! $keep_sources ]]
  then
    mkdir $directory
    git clone --bare git://github.com/$organisation/$project $directory/.git
    cd $directory
    git config --local --bool core.bare false
    cd ..
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  else
    git clone git://github.com/$organisation/$project $directory
    cd $directory
    git checkout $version
    cd ..
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  fi
}

function download_from_cmicdev() {
  project=$1
  version=$2
  directory=$project-$version
  tarball=$directory.tar.gz
  if [[ $keep_repository && ! $keep_sources ]]
  then
    mkdir $directory
    git clone --bare git://cmicdev.cs.ucl.ac.uk/$project $directory/.git
    cd $directory
    git config --local --bool core.bare false
    cd ..
  else
    git clone git://cmicdev.cs.ucl.ac.uk/$project $directory
    cd $directory
    git checkout $version
    if ! $keep_repository
    then
      rm -rf .git
    fi
    cd ..
  fi
  rm $tarball 2> /dev/null
  tar cvfz $tarball $directory
  rm -rf $directory
}

function download_from_sourceforge_svn() {
  project=$1
  version=$2
  path=$3
  project_lowercase=$(echo $project | tr [:upper:] [:lower:])
  # This was the old URI before SourceForge moved every project to the new server.
  #repo_url=https://$project_lowercase.svn.sourceforge.net/svnroot/$project_lowercase/$path
  repo_url=http://svn.code.sf.net/p/$project_lowercase/code/$path
  directory=$project-$version
  tarball=$directory.tar.gz
  if [ ! $keep_sources ]
  then
    print_usage
  fi
  if [ ! $keep_repository ]
  then
    svn export -r ${version} $repo_url $directory
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  else
    svn checkout -r ${version} $repo_url $directory
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  fi
}

function download_from_sourceforge_git() {
  project=$1
  version=$2
  directory=$project-$version
  project_lowercase=$(echo $project | tr [:upper:] [:lower:])
  tarball=$directory.tar.gz
  if [[ $keep_repository && ! $keep_sources ]]
  then
    mkdir $directory
    git clone --bare git://git.code.sf.net/p/$project_lowercase/code $directory/.git
    cd $directory
    git config --local --bool core.bare false
    cd ..
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  else
    git clone git://git.code.sf.net/p/$project_lowercase/code $directory
    cd $directory
    git checkout $version
    if [[ ! $keep_repository ]]
    then
      rm -rf .git
    fi
    cd ..
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  fi
}

if [[ $project = MITK || $project = OpenIGTLink ]]
then
  download_from_github NifTK $project $version
elif [ $project = commontk/CTK ]
then
  download_from_github commontk CTK $version
elif [ $project = NifTK/CTK ]
then
  download_from_github NifTK CTK $version
elif [ $project = SlicerExecutionModel ]
then
  download_from_github Slicer $project $version
elif [ $project = qRestAPI ]
then
  download_from_github commontk $project $version
elif [ $project = NiftySeg ]
then
  download_from_sourceforge_svn $project $version 
elif [ $project = NiftyReg ]
then
  download_from_sourceforge_svn $project $version trunk/nifty_reg
elif [ $project = NiftySim ]
then
#  download_from_sourceforge_svn $project $version trunk/nifty_sim
  download_from_sourceforge_git $project $version
elif [ $project = NiftyRec ]
then
  download_from_sourceforge_svn $project $version 
elif [ $project = NiftyLink ]
then
  download_from_cmicdev $project $version
elif [ $project = NifTKData ]
then
  download_from_cmicdev $project $version
elif [ $project = IGSTK ]
then
  if [[ $version = IGSTK-* ]]
  then
    directory=$version
  else
    directory=$project-$version
  fi
  tarball=$directory.tar.gz
  if [[ $keep_repository && ! $keep_sources ]]
  then
    mkdir $directory
    cd $directory
    git clone --bare git://igstk.org/IGSTK.git .git
    git config --local --bool core.bare false
  else
    git clone git://igstk.org/IGSTK.git $directory
    cd $directory
    git checkout $version
    if ! $keep_repository
    then
      rm -rf .git
    fi
  fi
  cd ..
  rm $tarball
  tar cvfz $tarball $directory
  rm -rf $directory
else
  print_usage
fi

rm $tarball.md5 2> /dev/null
md5sum $tarball > $tarball.md5
chmod 664 $tarball
chmod 664 $tarball.md5
