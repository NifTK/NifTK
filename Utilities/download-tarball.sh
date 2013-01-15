#!/bin/bash

# Downloads external project dependencies of NifTK and creates a
# tarball and MD5 checksum.

function print_usage() {
  echo "
Usage:
        ./download-tarball.sh [ -d|--discard-repo ] <project> <version>

Supported projects:

    MITK
    CTK
    OpenIGTLink
    SlicerExecutionModel
    qRestAPI
    espakm/CTK
    NiftyRec
    NiftyReg
    NiftySeg
    NiftySim
    NiftyLink
    IGSTK
"
  exit 1
}


if [ $# -eq 0 ]
then
    print_usage
fi

if [[ $1 = -d || $1 = --discard-repo ]]
then
    discard_repo=true
    shift
else
    discard_repo=false
fi

if [ $# -ne 2 ]
then
    print_usage
fi

project=$1
version=$2

function download_from_github() {
    organisation=$1
    project=$2
    version=$3
    directory=$organisation-$project-$version
    tarball=$directory.tar.gz
    if $discard_repo
    then
        wget -O $tarball http://github.com/$organisation/$project/tarball/$version
    else
        git clone git://github.com/$organisation/$project $directory
        cd $directory
        git checkout $version
        cd ..
        tar cvfz $tarball ${directory}
        rm -rf $directory
    fi
}

function download_from_sourceforge() {
    project=$1
    version=$2
    path=$3
    project_lowercase=$(echo $project | tr [:upper:] [:lower:])
    repo_url=https://$project_lowercase.svn.sourceforge.net/svnroot/$project_lowercase/$path
    directory=$project-$version
    tarball=$directory.tar.gz
    if $discard_repo
    then
        svn export -r ${version} $repo_url $directory
        tar cvfz $tarball $directory
        rm -rf $directory
    else
        svn checkout -r ${version} $repo_url $directory
        tar cvfz $tarball $directory
        rm -rf $directory
    fi
}

if [[ $project = MITK || $project = OpenIGTLink ]]
then
    download_from_github NifTK $project $version
elif [ $project = CTK ]
then
    organisation=commontk
    directory=NifTK-$project-$version
    tarball=$directory.tar.gz
    if $discard_repo
    then
        wget -O $tarball http://github.com/$organisation/$project/tarball/$version
    else
        git clone git://github.com/$organisation/$project $directory
        cd $directory
        git checkout $version
        cd ..
        tar cvfz $tarball ${directory}
        rm -rf $directory
    fi
elif [ $project = SlicerExecutionModel ]
then
    download_from_github Slicer $project $version
elif [ $project = qRestAPI ]
then
    download_from_github commontk $project $version
elif [ $project = espakm/CTK ]
then
    download_from_github espakm CTK $version
elif [ $project = NiftySeg ]
then
    download_from_sourceforge $project $version 
elif [ $project = NiftyReg ]
then
    download_from_sourceforge $project $version trunk/nifty_reg
elif [ $project = NiftySim ]
then
    download_from_sourceforge $project $version trunk/nifty_sim
elif [ $project = NiftyRec ]
then
    download_from_sourceforge $project $version 
elif [ $project = NiftyLink ]
then
    directory=$project-$version
    tarball=$directory.tar.gz
    git clone git://cmicdev.cs.ucl.ac.uk/$project $directory
    cd $directory
    git checkout $version
    if $discard_repo
    then
        rm -rf .git
    fi
    cd ..
    tar cvfz $tarball $directory
    rm -rf $directory
elif [ $project = IGSTK ]
then
    if [[ $version = IGSTK-* ]]
    then
        directory=$version
    else
        directory=$project-$version
    fi
    tarball=$directory.tar.gz
    git clone git://igstk.org/IGSTK.git $directory
    cd $directory
    git checkout $version
    if $discard_repo
    then
        rm -rf .git
    fi
    cd ..
    tar cvfz $tarball $directory
    rm -rf $directory
else
    print_usage
fi

md5sum $tarball > $tarball.md5
chmod 644 $tarball
chmod 644 $tarball.md5
