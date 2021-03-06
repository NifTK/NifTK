/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <niftkNiftyViewPreferencePage.h>
#include <niftkBaseApplicationPreferencePage.h>
#include "../niftkNiftyViewApplication.h"
#include "../niftkDnDDefaultPerspective.h"

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
  return QString("qthelp://uk.ac.ucl.cmic.niftyview/bundle/uk_ac_ucl_cmic_niftyview_intro.html");
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::start(context);

  BERRY_REGISTER_EXTENSION_CLASS(NiftyViewApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(DnDDefaultPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(NiftyViewPreferencePage, context);

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
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_niftyview, niftk::PluginActivator)
#endif
