/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPluginActivator_h
#define niftkPluginActivator_h

#include <ctkPluginActivator.h>

#include <niftkLookupTableProviderService.h>

namespace niftk
{

/**
 * \class PluginActivator
 * \brief CTK Plugin Activator class for ImageLookupTablesView.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class PluginActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_imagelookuptables")
#endif

public:

  PluginActivator();
  virtual ~PluginActivator();

  static PluginActivator* GetInstance();

  ctkPluginContext* GetContext() const;

  LookupTableProviderService* GetLookupTableProviderService() const;

  void start(ctkPluginContext* context) override;

  void stop(ctkPluginContext* context) override;

private:

  ctkPluginContext* m_Context;

  static PluginActivator* s_Instance;

};

}

#endif
