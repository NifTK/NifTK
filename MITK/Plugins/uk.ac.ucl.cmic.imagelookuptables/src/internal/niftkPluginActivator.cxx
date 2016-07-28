/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <QtPlugin>

#include "niftkImageLookupTablesView.h"
#include "niftkImageLookupTablesPreferencePage.h"

namespace niftk
{

PluginActivator* PluginActivator::s_Instance = nullptr;

//-----------------------------------------------------------------------------
PluginActivator::PluginActivator()
: m_Context(nullptr)
{
  assert(!s_Instance);
  s_Instance = this;
}


//-----------------------------------------------------------------------------
PluginActivator::~PluginActivator()
{
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  m_Context = context;

  BERRY_REGISTER_EXTENSION_CLASS(ImageLookupTablesView, context);
  BERRY_REGISTER_EXTENSION_CLASS(ImageLookupTablesPreferencePage, context);
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
PluginActivator* PluginActivator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PluginActivator::GetContext() const
{
  return m_Context;
}


//-----------------------------------------------------------------------------
LookupTableProviderService* PluginActivator::GetLookupTableProviderService() const
{
  ctkServiceReference serviceRef = m_Context->getServiceReference<LookupTableProviderService>();
  LookupTableProviderService* lutService = m_Context->getService<LookupTableProviderService>(serviceRef);

  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find niftk::LookupTableProviderService." << std::endl;
  }

  return lutService;
}

}

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_imagelookuptables, niftk::PluginActivator)
#endif
