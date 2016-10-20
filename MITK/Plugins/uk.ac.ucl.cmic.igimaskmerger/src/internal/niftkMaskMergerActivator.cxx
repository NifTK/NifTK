/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMaskMergerActivator.h"
#include "niftkMaskMergerView.h"

#include <QtPlugin>

namespace niftk
{

ctkPluginContext* MaskMergerActivator::m_PluginContext = nullptr;

//-----------------------------------------------------------------------------
void MaskMergerActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(MaskMergerView, context);
  m_PluginContext = context;
}

//-----------------------------------------------------------------------------
void MaskMergerActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ctkPluginContext* MaskMergerActivator::getContext()
{
  return m_PluginContext;
}


} // end namespace

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_igimaskmerger, niftk::MaskMergerActivator)
#endif
