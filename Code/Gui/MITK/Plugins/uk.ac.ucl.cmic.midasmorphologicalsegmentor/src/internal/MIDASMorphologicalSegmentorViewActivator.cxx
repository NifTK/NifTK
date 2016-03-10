/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASMorphologicalSegmentorViewActivator.h"
#include "MIDASMorphologicalSegmentorView.h"
#include <QtPlugin>
#include "MIDASMorphologicalSegmentorViewPreferencePage.h"

namespace niftk
{

//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(MIDASMorphologicalSegmentorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(MIDASMorphologicalSegmentorViewPreferencePage, context);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_midasmorphologicalsegmentor, niftk::MIDASMorphologicalSegmentorViewActivator)
#endif
