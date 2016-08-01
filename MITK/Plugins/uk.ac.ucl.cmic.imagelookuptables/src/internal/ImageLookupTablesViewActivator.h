/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ImageLookupTablesViewActivator_h
#define ImageLookupTablesViewActivator_h

#include <ctkPluginActivator.h>
#include <QmitkLookupTableProviderService.h>

namespace mitk {

/**
 * \class ImageLookupTablesViewActivator
 * \brief CTK Plugin Activator class for ImageLookupTablesView.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 */
class ImageLookupTablesViewActivator :
  public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_imagelookuptables")
#endif

public:

  ImageLookupTablesViewActivator();
  ~ImageLookupTablesViewActivator();

  static ImageLookupTablesViewActivator* GetDefault();
  ctkPluginContext* GetPluginContext() const;
  static QmitkLookupTableProviderService* GetQmitkLookupTableProviderService();

  void start(ctkPluginContext* context) override;
  void stop(ctkPluginContext* context) override;

private:

  ctkPluginContext* m_Context;
  static ImageLookupTablesViewActivator* s_Inst;
};

}

#endif