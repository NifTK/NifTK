/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "../NifTKApplication.h"
/**
 * \file NiftyIGI.cpp
 * \brief Main entry point for NiftyIGI application.
 */
int main(int argc, char** argv)
{
  return ApplicationMain(argc, argv, "NiftyIGI", "CMIC", "uk.ac.ucl.cmic.gui.qt.niftyigi", "liborg_mitk_gui_qt_ext");
}
