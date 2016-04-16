/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCommonMIDASActivator_h
#define niftkCommonMIDASActivator_h

#include <ctkPluginActivator.h>

namespace niftk
{

class CommonMIDASActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context) override;

  void stop(ctkPluginContext* context) override;

};

}

#endif
