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

namespace mitk {

//-----------------------------------------------------------------------------
void ImageLookupTablesViewActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(ImageLookupTablesView, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkImageLookupTablesPreferencePage, context);
}


//-----------------------------------------------------------------------------
void ImageLookupTablesViewActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_imagelookuptables, mitk::ImageLookupTablesViewActivator)
