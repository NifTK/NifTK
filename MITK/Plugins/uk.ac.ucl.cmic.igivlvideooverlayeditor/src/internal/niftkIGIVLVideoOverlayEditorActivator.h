/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIVLVideoOverlayEditorActivator_h
#define niftkIGIVLVideoOverlayEditorActivator_h

#include <ctkPluginActivator.h>

namespace niftk
{

/**
 * \class IGIVLVideoOverlayEditorActivator
 * \brief Activator class for the IGIVLVideoOverlayEditor.
 * \ingroup uk_ac_ucl_cmic_IGIVLVideoOverlayeditor_internal
 */
class IGIVLVideoOverlayEditorActivator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_igivlvideooverlayeditor")
#endif

public:

  void start(ctkPluginContext* context) override;
  void stop(ctkPluginContext* context) override;
  static ctkPluginContext* getContext();

private:
  static ctkPluginContext* m_PluginContext;

};

} // end namespace

#endif

