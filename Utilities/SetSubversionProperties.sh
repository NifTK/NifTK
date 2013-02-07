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

echo ""
echo "Setting svn:eol-style and svn:keywords"
echo ""

for f in h c cpp cxx txx txt sh csh sh.in csh.in
do
  echo "Checking for:${f}"
  find . -name "*.${f}" -exec svn propset svn:eol-style native {} \;
  find . -name "*.${f}" -exec svn propset svn:keywords "Date Revision Author HeadURL Rev" {} \;
done

echo ""
echo "Setting svn:executable"
echo ""

for f in sh csh sh.in csh.in
do
  echo "Checking for:${f}"
  find . -name "*.${f}" -exec svn propset svn:executable "" {} \;
done