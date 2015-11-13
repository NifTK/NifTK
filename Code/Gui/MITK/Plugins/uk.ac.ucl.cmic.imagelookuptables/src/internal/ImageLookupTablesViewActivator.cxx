/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ImageLookupTablesViewActivator.h"
#include "ImageLookupTablesView.h"
#include <QtPlugin>

#include "QmitkImageLookupTablesPreferencePage.h"

namespace mitk 
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
  BERRY_REGISTER_EXTENSION_CLASS(QmitkImageLookupTablesPreferencePage, context);
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
QmitkLookupTableProviderService* ImageLookupTablesViewActivator::GetQmitkLookupTableProviderService()
{
  ctkPluginContext* context = ImageLookupTablesViewActivator::GetDefault()->GetPluginContext();
  ctkServiceReference serviceRef = context->getServiceReference<QmitkLookupTableProviderService>();
  QmitkLookupTableProviderService* lutService = context->getService<QmitkLookupTableProviderService>(serviceRef);
  
  if (lutService == NULL)
  {
    mitkThrow() << "Failed to find QmitkLookupTableProviderService." << std::endl;
  }

  return lutService;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_imagelookuptables, mitk::ImageLookupTablesViewActivator)
