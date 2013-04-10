/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASGeneralSegmentorViewActivator.h"
#include "MIDASGeneralSegmentorView.h"
#include <QtPlugin>
#include "MIDASGeneralSegmentorViewPreferencePage.h"

namespace mitk {

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(MIDASGeneralSegmentorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(MIDASGeneralSegmentorViewPreferencePage, context);
}

//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_midasgeneralsegmentor, mitk::MIDASGeneralSegmentorViewActivator)
