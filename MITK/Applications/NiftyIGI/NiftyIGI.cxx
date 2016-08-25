/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <QVariant>

/// \file NiftyIGI.cxx
/// \brief Main entry point for NiftyIGI application.
int main(int argc, char** argv)
{
  niftk::BaseApplication app(argc, argv);
  app.setApplicationName("NiftyIGI");
  app.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftyigi");

  return app.run();
}
