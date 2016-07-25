/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLStandardDisplayEditorActivator_h
#define niftkVLStandardDisplayEditorActivator_h

#include <ctkPluginActivator.h>

namespace niftk
{

/**
 * \class VLStandardDisplayEditorActivator
 * \brief Activator class for the VLStandardDisplayEditor.
 * \ingroup uk_ac_ucl_cmic_VLStandardDisplayeditor_internal
 */
class VLStandardDisplayEditorActivator : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_vlstandarddisplayeditor")
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

