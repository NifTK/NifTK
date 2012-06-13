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

#include "SurfaceExtractorPluginActivator.h"

#include "SurfaceExtractorView.h"
#include "SurfaceExtractorPreferencePage.h"

#include <QtPlugin>

namespace mitk {

void SurfaceExtractorPluginActivator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceExtractorView, context);
  BERRY_REGISTER_EXTENSION_CLASS(SurfaceExtractorPreferencePage, context);
}

void SurfaceExtractorPluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_surfaceextractor, mitk::SurfaceExtractorPluginActivator)
