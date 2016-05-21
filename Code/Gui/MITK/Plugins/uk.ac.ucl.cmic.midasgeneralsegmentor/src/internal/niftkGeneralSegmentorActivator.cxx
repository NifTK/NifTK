/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorActivator.h"
#include "niftkGeneralSegmentorView.h"
#include <QtPlugin>
#include "niftkGeneralSegmentorPreferencePage.h"

#include <niftkTool.h>

namespace niftk
{

//-----------------------------------------------------------------------------
void GeneralSegmentorActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(GeneralSegmentorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(GeneralSegmentorPreferencePage, context);

  MIDASTool::LoadBehaviourStrings();
}

//-----------------------------------------------------------------------------
void GeneralSegmentorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_midasgeneralsegmentor, niftk::GeneralSegmentorActivator)
#endif
