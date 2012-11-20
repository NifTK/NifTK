/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8038 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "../NifTKApplication.h"
/**
 * \file NiftyMIDAS.cpp
 * \brief Main entry point for NiftyMIDAS application.
 */
int main(int argc, char** argv)
{
  return ApplicationMain(argc, argv, "NiftyMIDAS", "CMIC", "uk.ac.ucl.cmic.gui.qt.niftymidas", "liborg_mitk_gui_qt_ext");
}
