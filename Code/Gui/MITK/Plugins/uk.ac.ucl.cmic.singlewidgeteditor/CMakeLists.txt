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

project(uk_ac_ucl_cmic_singlewidgeteditor)

MACRO_CREATE_MITK_CTK_PLUGIN(
	EXPORT_DIRECTIVE UK_AC_UCL_CMIC_SINGLEWIDGETEDITOR
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES Qmitk niftkQmitkExt
  SUBPROJECTS MITK-CoreUI
  )
