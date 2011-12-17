#!/bin/bash

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
#  Last Changed      : $LastChangedDate: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $ 
#  Revision          : $Revision: 6322 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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