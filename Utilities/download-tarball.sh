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

    download-tarball.sh [-s|--sources] [-r|--repository] [-h|--history] <project> <version>

Options:

    -s, --sources         Put the source files in the tarball.

    -r, --repository      Put the files of the versioning system in the tarball.

    -h, --history         Keep whole the revision history in the tarball.

The options can be combined but at least one of them must be given. The suggested
options are -r for the git and -sr for the subversion projects.

The '--history' option implies '--repository' and it can be applied for projects
with a git repository. With this option the tarball will contain the whole cloned
repository, with the complete history of every remote branches. When this option
is omitted (but '--repository' is specified), the tarball will only contain the
commit that is specified in the '<version>' argument.

Keeping the repository without history requires git 2.5.0 or newer.
Commands to install git into your home bin directory:

    git clone https://github.com/git/git
    cd git
    make
    make install

Supported projects:

    MITK
    EpiNav-MITK
    commontk/CTK
    NifTK/CTK
    apriltags
    pcl
    OpenIGTLink
    SlicerExecutionModel
    qRestAPI
    commontk/PythonQt
    NifTK/PythonQt
    NiftyRec
    NiftyReg
    EpiNav-NiftyReg
    NiftySeg
    NiftySim
    NiftyLink
    NifTKData
    IGSTK
    NifTK
    RTK
    camino
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
  elif [[ $1 = -h || $1 = --history || $1 = -rh || $1 = -hr ]]
  then
    keep_repository=true
    keep_history=true
    shift
  elif [[ $1 = -sr || $1 = -rs ]]
  then
    keep_sources=true
    keep_repository=true
    shift
  elif [[ $1 = -sh || $1 = -hs || $1 = -srh || $1 = -shr || $1 = -rsh || $1 = -rhs || $1 = -hsr || $1 = -hrs ]]
  then
    keep_sources=true
    keep_repository=true
    keep_history=true
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

if [[ $keep_repository && ! $keep_history ]]
then
  git_version_regex="^git version ([0-9]+)\.([0-9]+)\.([0-9]+)(\.([0-9]+))?$"
  git_version=`git --version`
  if [[ $git_version =~ $git_version_regex ]]
  then
    git_major_version="${BASH_REMATCH[1]}"
    git_minor_version="${BASH_REMATCH[2]}"
    if [[ ! ( ${git_major_version} -ge 2 && ${git_minor_version} -ge 5 ) ]]
    then
      echo "Obsolete git version."
      print_usage
    fi
  else
    echo "Unrecognised git version."
    print_usage
  fi
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
  elif [[ ! $keep_sources && $keep_history ]]
  then
    mkdir $directory
    git clone --bare git://github.com/$organisation/$project $directory/.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git@github.com:$organisation/$project
    cd ..
    rm $tarball 2> /dev/null
    tar cvfz $tarball $directory
    rm -rf $directory
  elif [[ ! $keep_sources && $keep_repository ]]
  then
    mkdir $directory
    git clone --bare git://github.com/$organisation/$project $directory/.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    version_sha1=`git rev-parse $version`
    git config uploadpack.allowReachableSHA1InWant true
    cd ..
    mkdir ${directory}-shallow
    cd ${directory}-shallow
    git init
    git fetch --depth=1 ../$directory $version_sha1
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git@github.com:$organisation/$project
    cd ..
    rm -rf $directory
    mv $directory-shallow $directory
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

function download_from_cmiclab() {
  project=$1
  version=$2
  directory=$project-$version
  tarball=$directory.tar.gz
  if [[ $keep_history || ! $keep_repository ]]
  then
    mkdir $directory
    git clone --bare git@cmiclab.cs.ucl.ac.uk:CMIC/$project $directory/.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git@cmiclab.cs.ucl.ac.uk:CMIC/$project
    if [[ $keep_sources ]]
    then
      git checkout $version
    fi
    if [ ! $keep_repository ]
    then
      rm -rf .git
    fi
    cd ..
  else
    mkdir $directory
    git clone --bare git@cmiclab.cs.ucl.ac.uk:CMIC/$project $directory/.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    version_sha1=`git rev-parse $version`
    git config uploadpack.allowReachableSHA1InWant true
    cd ..
    mkdir ${directory}-shallow
    cd ${directory}-shallow
    git init
    git fetch --depth=1 ../$directory $version_sha1
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git@cmiclab.cs.ucl.ac.uk:CMIC/$project
    if [[ $keep_sources ]]
    then
        git checkout $version_sha1
    fi
    cd ..
    rm -rf $directory
    mv $directory-shallow $directory
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
  suffix=$3
  directory=$project-$version
  project_lowercase=$(echo $project | tr [:upper:] [:lower:])
  tarball=$directory.tar.gz
  if [[ $keep_history || ! $keep_repository ]]
  then
    mkdir $directory
    git clone --bare git://git.code.sf.net/p/$project_lowercase/$suffix $directory/.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git://git.code.sf.net/p/$project_lowercase/$suffix
    if [[ $keep_sources ]]
    then
      git checkout $version
    fi
    if [ ! $keep_repository ]
    then
      rm -rf .git
    fi
    cd ..
  else
    mkdir $directory
    git clone git://git.code.sf.net/p/$project_lowercase/$suffix $directory.git
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    version_sha1=`git rev-parse $version`
    git config uploadpack.allowReachableSHA1InWant true
    cd ..
    mkdir ${directory}-shallow
    cd ${directory}-shallow
    git init
    git fetch --depth=1 ../$directory $version_sha1
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
    git remote add origin git://git.code.sf.net/p/$project_lowercase/$suffix
    if [[ $keep_sources ]]
    then
        git checkout $version_sha1
    fi
    cd ..
    rm -rf $directory
    mv $directory-shallow $directory
  fi
  rm $tarball 2> /dev/null
  tar cvfz $tarball $directory
  rm -rf $directory
}

if [[ $project = MITK || $project = OpenIGTLink || $project = apriltags ]]
then
  download_from_github NifTK $project $version
elif [[ $project = "EpiNav-MITK" ]]
then
  download_from_cmiclab $project $version
elif [ $project = commontk/CTK ]
then
  download_from_github commontk CTK $version
elif [ $project = NifTK/CTK ]
then
  download_from_github NifTK CTK $version
elif [ $project = SlicerExecutionModel ]
then
  download_from_github Slicer $project $version
elif [ $project = pcl ]
then
  download_from_github PointCloudLibrary $project $version
elif [ $project = RTK ]
then
  download_from_github NifTK $project $version
elif [ $project = qRestAPI ]
then
  download_from_github commontk $project $version
elif [ $project = commontk/PythonQt ]
then
  download_from_github commontk PythonQt $version
elif [ $project = NifTK/PythonQt ]
then
  download_from_github NifTK PythonQt $version
elif [ $project = NiftySeg ]
then
  download_from_sourceforge_git $project $version git
elif [ $project = NiftyReg ]
then
  download_from_sourceforge_git $project $version git
elif [ $project = "EpiNav-NiftyReg" ]
then
  download_from_cmiclab $project $version
elif [ $project = NiftySim ]
then
  download_from_sourceforge_git $project $version code
elif [ $project = NiftyRec ]
then
  download_from_sourceforge_svn $project $version
elif [ $project = camino ]
then
  download_from_sourceforge_git $project $version git
elif [ $project = NiftyLink ]
then
  download_from_cmiclab $project $version
elif [ $project = NifTKData ]
then
  download_from_cmiclab $project $version
elif [ $project = NifTK ]
then
  download_from_cmiclab $project $version
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
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
    git clone --bare git://igstk.org/IGSTK.git .git
    git config --local --bool core.bare false
    mkdir -p .git/logs/refs
  else
    git clone git://igstk.org/IGSTK.git $directory
    cd $directory
    if ! git cat-file -e $version 2> /dev/null
    then
        echo "The commit does not exist in the repository."
        cd ..
        rm -rf $directory
        exit 1
    fi
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
