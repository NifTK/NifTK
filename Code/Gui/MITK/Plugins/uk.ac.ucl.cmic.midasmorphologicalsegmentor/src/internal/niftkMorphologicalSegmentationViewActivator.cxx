/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentationViewActivator.h"
#include "niftkMorphologicalSegmentationView.h"
#include <QtPlugin>
#include "niftkMorphologicalSegmentationViewPreferencePage.h"

namespace niftk
{

//-----------------------------------------------------------------------------
void MorphologicalSegmentationViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(niftkMorphologicalSegmentationView, context);
  BERRY_REGISTER_EXTENSION_CLASS(niftkMorphologicalSegmentationViewPreferencePage, context);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentationViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_midasmorphologicalsegmentor, niftk::MorphologicalSegmentationViewActivator)
#endif
