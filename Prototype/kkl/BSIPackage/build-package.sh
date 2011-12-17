#!/bin/bash 
set -u 

if [ $# \< 1 ]
then
  echo "Usage: $0 version"
  echo "  version : version number of the package"
  exit
fi 

version=$1
project_name=KN-BSI-${version}

rm -rf ${project_name}
rm -f ${project_name}.zip
mkdir ${project_name}

# Copy all the C++ template files. 
cp ../../../Code/Ext/ITK/BoundaryShiftIntegral/*.h ${project_name}/.
cp ../../../Code/Ext/ITK/BoundaryShiftIntegral/*.txx ${project_name}/.

# Copy the two applications. 
# cp ../../../Code/Applications/niftkBSI.cxx ${project_name}/.
cp ../../../Code/Applications/niftkKMeansWindowWithLinearRegressionNormalisationBSI.cxx ${project_name}/.

cp CMakeLists.txt ${project_name}/.

cp SoftwareLicence.txt ${project_name}/.
cp Readme.txt ${project_name}/.

zip -r ${project_name}.zip ${project_name}

rm -rf ${project_name}
