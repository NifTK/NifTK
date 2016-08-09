/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegmentorActivator.h"
#include "niftkCaffeSegmentorView.h"
#include "niftkCaffeSegmentorPreferencePage.h"

#include <QtPlugin>

namespace niftk
{

ctkPluginContext* CaffeSegmentorActivator::m_PluginContext = nullptr;

//-----------------------------------------------------------------------------
void CaffeSegmentorActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(CaffeSegmentorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(CaffeSegmentorPreferencePage, context);
  m_PluginContext = context;
}

//-----------------------------------------------------------------------------
void CaffeSegmentorActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* CaffeSegmentorActivator::getContext()
{
  return m_PluginContext;
}


} // end namespace

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_caffesegmentor, niftk::CaffeSegmentorActivator)
#endif
