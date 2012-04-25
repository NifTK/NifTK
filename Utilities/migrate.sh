#!/bin/bash
THIS_SCRIPT=$(realpath ${0})

#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-10-27 11:01:03 +0100 (Thu, 27 Oct 2011) $ 
#  Revision          : $Revision: 7614 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : stian.johnsen.09@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

function _rename_files() {
    local OLD_SUBSTR=${1}
    local NEW_SUBSTR=${2}
    local FILE=""
    local NEW_FILE_NAME=""

    echo "Looking for files matching *${OLD_SUBSTR}*"
    for FILE in *${OLD_SUBSTR}*; do 	
	if [ ! -f ${FILE} -a ! -d ${FILE} ]; then 
	    echo "No suitable files found exiting"
	    break; 
	fi
	NEW_FILE_NAME=${FILE/${OLD_SUBSTR}/${NEW_SUBSTR}}
	echo "Renaming ${FILE} to ${NEW_FILE_NAME}"
	mv ${FILE} ${NEW_FILE_NAME}
    done    
}

if [ "x${1}" == "x-r" ]; then
    OLD_NAME=${2}
    NEW_NAME=${3}
    DIR=${4}

    if [ -z "${OLD_NAME}" ] || [ -z "${NEW_NAME}" ]; then
	echo "Usage: ${THIS_FUNCTION} <OLD IDENTIFIER> <NEW IDENTIFIER> <DIRECTORY>" 1>&2
	exit 1
    fi

    if [ ! -z "${DIR}" ]; then
	cd ${DIR} || exit 1
    else
	echo "Error no directory specified. Exiting..." 1>&2	
	exit 1
    fi   

    echo "Changing ${OLD_NAME} to ${NEW_NAME} in ${PWD}"
    _rename_files ${OLD_NAME} ${NEW_NAME}

    for DIRITEM in *; do
	if [ $(basename ${DIRITEM}) == $(basename ${THIS_SCRIPT}) ]; then
	    continue
	fi

	if [ -d ${DIRITEM} -a "${DIRITEM}" != ".." -a "${DIRITEM}" != "." -a "${DIRITEM}" != ".svn" -a "${DIRITEM}" != "CVS" ]; then	
	    pwd
	    echo "Descending into ${DIRITEM}:"
	    echo "Renaming ${OLD_NAME} -> ${NEW_NAME}"
	    ${THIS_SCRIPT} -r ${OLD_NAME} ${NEW_NAME} ${DIRITEM}
	else
	    if [ -f ${DIRITEM} ]; then
		cp ${DIRITEM} ${DIRITEM}.bk
		echo "sed -r 's/${OLD_NAME}/${NEW_NAME}/g' ${DIRITEM}.bk > ${DIRITEM}"
		sed -r "s/${OLD_NAME}/${NEW_NAME}/g" ${DIRITEM}.bk > ${DIRITEM}
		rm -f ${DIRITEM}.bk
	    fi
	fi
    done
elif [ "x${1}" == "x-h" -o "x${1}" == "x-help" -o "x${1}" == "x--help" ]; then
    echo -e "USAGE: ./migrate.sh\n\tExports a clean copy of UCLTK to a directory called NifTK, and subsequently substitutes ucltk-type strings for niftk ones (including file names)."
    exit 0
else
# START OF SCRIPT
    svn export https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/UCLToolkit NifTK
    if [ $? != 0 ]; then
	echo "Export failed" 1>&2
	exit 1
    fi

    ${THIS_SCRIPT} -r ucltk niftk NifTK
    ${THIS_SCRIPT} -r UCLTK NIFTK NifTK
    ${THIS_SCRIPT} -r UCLTOOLKIT NIFTK NifTK
    ${THIS_SCRIPT} -r UCLToolkit NifTK NifTK
fi