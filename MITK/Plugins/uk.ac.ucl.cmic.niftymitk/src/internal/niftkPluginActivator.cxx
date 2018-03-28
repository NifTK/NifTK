/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <niftkMinimalPerspective.h>
#include <niftkIGIPerspective.h>
#include <niftkNiftyMITKPreferencePage.h>
#include <niftkBaseApplicationPreferencePage.h>
#include "../niftkNiftyMITKApplication.h"


namespace niftk
{

//-----------------------------------------------------------------------------
PluginActivator::PluginActivator()
{
}


//-----------------------------------------------------------------------------
PluginActivator::~PluginActivator()
{
}


//-----------------------------------------------------------------------------
QString PluginActivator::GetHelpHomePageURL() const
{
  return QString("qthelp://uk.ac.ucl.cmic.niftyview/bundle/uk_ac_ucl_cmic_niftymitk_intro.html");
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::start(context);

  BERRY_REGISTER_EXTENSION_CLASS(NiftyMITKApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(MinimalPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(IGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(NiftyMITKPreferencePage, context);

  this->RegisterHelpSystem();
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::stop(context);
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_niftymitk, niftk::PluginActivator)
#endif
