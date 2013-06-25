#!/bin/bash

#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

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
    NifTKData
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
        rm $tarball
        wget -O $tarball http://github.com/$organisation/$project/tarball/$version
    else
        git clone git://github.com/$organisation/$project $directory
        cd $directory
        git checkout $version
        cd ..
        rm $tarball
        tar cvfz $tarball ${directory}
        rm -rf $directory
    fi
}

# Requires a specific commit hash -> !
function download_from_sf_git() {
    project=${1}
    path=${2}
    commit=${3}    
    branch=${4}
    local LC_PROJNAME=`echo ${project} | tr [:upper:] [:lower:]`
    local CI_FS_ID=`echo ${commit} | head -c 4`
    local DST_PATH=${LC_PROJNAME}-${CI_FS_ID}

    if [ ! -d ${DST_PATH} ]; then
	local GIT_CMD="git clone "

	if [ "x${branch}" != "x" ]; then
	    GIT_CMD="${GIT_CMD} -b \"${branch}\" "
	fi
    
	eval "${GIT_CMD} git://git.code.sf.net/p/${LC_PROJNAME}/code ${DST_PATH}" 
    fi

    pushd ${DST_PATH}
    git reset --hard ${commit}
    popd

    tar zcf ${DST_PATH}{.tar.gz,}
    
    tarball=${DST_PATH}.tar.gz
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
        rm $tarball
        tar cvfz $tarball $directory
        rm -rf $directory
    else
        svn checkout -r ${version} $repo_url $directory
        rm $tarball
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
        rm $tarball
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
    download_from_sf_git $project niftysim-2.0 $version
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
    rm $tarball
    tar cvfz $tarball $directory
    rm -rf $directory
elif [ $project = NifTKData ]
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
    rm $tarball
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
    rm $tarball
    tar cvfz $tarball $directory
    rm -rf $directory
else
    print_usage
fi
    
[ -f $tarball.md5 ] && rm $tarball.md5
md5sum $tarball > $tarball.md5
chmod 664 $tarball
chmod 664 $tarball.md5
