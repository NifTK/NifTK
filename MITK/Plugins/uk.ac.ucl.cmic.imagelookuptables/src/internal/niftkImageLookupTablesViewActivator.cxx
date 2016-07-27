/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkImageLookupTablesViewActivator.h"

#include <QtPlugin>

#include "niftkImageLookupTablesView.h"
#include "niftkImageLookupTablesPreferencePage.h"

namespace niftk
{

ImageLookupTablesViewActivator* ImageLookupTablesViewActivator::s_Inst = 0;

//-----------------------------------------------------------------------------
ImageLookupTablesViewActivator::ImageLookupTablesViewActivator()
: m_Context(NULL)
{
  s_Inst = this;
}


//-----------------------------------------------------------------------------
ImageLookupTablesViewActivator::~ImageLookupTablesViewActivator()
{
}


//-----------------------------------------------------------------------------
void ImageLookupTablesViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(ImageLookupTablesView, context);
  BERRY_REGISTER_EXTENSION_CLASS(ImageLookupTablesPreferencePage, context);
  m_Context = context;
}


//-----------------------------------------------------------------------------
void ImageLookupTablesViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
ImageLookupTablesViewActivator* ImageLookupTablesViewActivator::GetDefault()
{
  return s_Inst;
}


//-----------------------------------------------------------------------------
ctkPluginContext* ImageLookupTablesViewActivator::GetPluginContext() const
{
  return m_Context;
}


//-----------------------------------------------------------------------------
LookupTableProviderService* ImageLookupTablesViewActivator::GetLookupTableProviderService()
{
  ctkPluginContext* context = ImageLookupTablesViewActivator::GetDefault()->GetPluginContext();
  ctkServiceReference serviceRef = context->getServiceReference<LookupTableProviderService>();
  LookupTableProviderService* lutService = context->getService<LookupTableProviderService>(serviceRef);
  
  if (lutService == NULL)
  {
    mitkThrow() << "Failed to find niftk::LookupTableProviderService." << std::endl;
  }

  return lutService;
}

}

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_imagelookuptables, mitk::ImageLookupTablesViewActivator)
#endif
