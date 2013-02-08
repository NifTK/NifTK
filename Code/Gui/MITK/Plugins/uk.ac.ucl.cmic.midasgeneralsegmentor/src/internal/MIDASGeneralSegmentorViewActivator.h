/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGENERALSEGMENTORVIEWACTIVATOR_H
#define MIDASGENERALSEGMENTORVIEWACTIVATOR_H

#include <ctkPluginActivator.h>

namespace mitk {

/**
 * \class MIDASGeneralSegmentorViewActivator
 * \brief CTK Plugin Activator class for MIDASGeneralSegmentorView.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class MIDASGeneralSegmentorViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)

public:

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

}; // MIDASGeneralSegmentorViewActivator

}

#endif // MIDASGENERALSEGMENTORVIEWACTIVATOR_H
