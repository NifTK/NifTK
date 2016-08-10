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


namespace niftk
{

class NiftyMIDAS : public BaseApplication
{
public:
  NiftyMIDAS(int argc, char **argv)
    : BaseApplication(argc, argv)
  {
    this->setApplicationName("NiftyMIDAS");
    this->setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftymidas");
  }
};

}

/// \file NiftyMIDAS.cxx
/// \brief Main entry point for NiftyMIDAS application.
int main(int argc, char** argv)
{
  niftk::NiftyMIDAS app(argc, argv);

  return app.run();
}
