/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
