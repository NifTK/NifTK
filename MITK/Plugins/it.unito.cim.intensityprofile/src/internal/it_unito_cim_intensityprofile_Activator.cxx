/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "it_unito_cim_intensityprofile_Activator.h"

#include <QtPlugin>

#include "IntensityProfileView.h"
#include "PropagateSegmentationAlongTimeAction.h"

namespace mitk {

void it_unito_cim_intensityprofile_Activator::start(ctkPluginContext* context)
{
  BERRY_REGISTER_EXTENSION_CLASS(IntensityProfileView, context);
  BERRY_REGISTER_EXTENSION_CLASS(PropagateSegmentationAlongTimeAction, context);
}

void it_unito_cim_intensityprofile_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(it_unito_cim_intensityprofile, mitk::it_unito_cim_intensityprofile_Activator)
#endif