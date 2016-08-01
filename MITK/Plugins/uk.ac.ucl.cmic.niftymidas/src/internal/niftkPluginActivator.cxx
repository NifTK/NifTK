/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <niftkSegmentationPerspective.h>
#include <niftkQCPerspective.h>
#include <niftkBaseApplicationPreferencePage.h>
#include "../niftkNiftyMIDASApplication.h"


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
  return QString("qthelp://uk.ac.ucl.cmic.niftymidas/bundle/uk_ac_ucl_cmic_niftymidas_intro.html");
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::start(context);

  BERRY_REGISTER_EXTENSION_CLASS(NiftyMIDASApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(SegmentationPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QCPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(BaseApplicationPreferencePage, context);

  this->RegisterHelpSystem();
  /// Note:
  /// By default there is a global reinit after file open what reinitialises the global
  /// rendering manager. The ideal would be if the DnD Display could use its own rendering
  /// manager (RM), not the global one. This, however, does not work now because many MITK
  /// views have hard coded reference to the global RM, and they call RequestUpdate on that,
  /// not on the RM of the focused renderer. Until this is fixed in MITK, we have to suppress
  /// the global reinit after file open, and should not use the MITK Display and the DnD Display
  /// together in the same application.
  this->SetFileOpenTriggersReinit(false);
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::stop(context);
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_niftymidas, niftk::PluginActivator)
#endif
