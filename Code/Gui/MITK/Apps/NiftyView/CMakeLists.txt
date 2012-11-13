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
#  Last Changed      : $LastChangedDate: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
PROJECT(NiftyView)
SET(MY_APP_NAME NiftyView)

SET(_exclude_plugins
  uk_ac_ucl_cmic_breastsegmentation
  uk_ac_ucl_cmic_gui_qt_niftyigi
  uk_ac_ucl_cmic_gui_qt_niftymidas
  uk_ac_ucl_cmic_niftyseg
)

NIFTK_CREATE_APPLICATION(
  NAME NiftyView
  EXCLUDE_PLUGINS ${_exclude_plugins}
)
