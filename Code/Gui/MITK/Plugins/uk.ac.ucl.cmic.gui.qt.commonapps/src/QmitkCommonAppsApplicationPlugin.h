/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_
#define QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <ctkPluginActivator.h>
#include <ctkServiceTracker.h>
#include <berryIPreferencesService.h>
#include <mitkIDataStorageService.h>

#include <QObject>
#include <QString>

namespace mitk {
  class DataNode;
  class DataStorage;
}

/**
 * \class QmitkCommonAppsApplicationPlugin
 * \brief Abstract class that implements QT and CTK specific functionality to launch the application as a plugin.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps_internal
 */
class CMIC_QT_COMMONAPPS QmitkCommonAppsApplicationPlugin : public QObject, public ctkPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
  
public:

  QmitkCommonAppsApplicationPlugin();
  ~QmitkCommonAppsApplicationPlugin();

  static QmitkCommonAppsApplicationPlugin* GetDefault();
  ctkPluginContext* GetPluginContext() const;

  void start(ctkPluginContext* context);
  void stop(ctkPluginContext* context);

protected:

  /// \brief Deliberately not virtual method that enables derived classes to set the plugin context, and should be called from within the plugin start method.
  void SetPluginContext(ctkPluginContext*);

  /// \brief Deliberately not virtual method that connects this class to DataStorage so that we can receive NodeAdded events etc.
  void RegisterDataStorageListener();

  /// \brief Deliberately not virtual method that ddisconnects this class from DataStorage so that we can receive NodeAdded events etc.
  void UnregisterDataStorageListener();

  /// \brief Deliberately not virtual method thats called by derived classes within the start method to set up the help system.
  void RegisterHelpSystem();

  /// \brief Deliberately not virtual method that registers the Global Interaction Patterns developed as part of the MIDAS project.
  void RegisterMIDASGlobalInteractionPatterns();

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial LevelWindow property to each image.
  void RegisterLevelWindowProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial value for Texture Interpolation, and Reslice Interpolation.
  void RegisterInterpolationProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial value for black opacity property.
  void RegisterBlackOpacityProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to register any extensions that this plugin knows about.
  void RegisterQmitkCommonAppsExtensions();

  /// \brief Deliberately not virtual method thats called by derived classes, to set the departmental logo to blank.
  void BlankDepartmentalLogo();

  /// \brief Called each time a data node is added, and derived classes can override it.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Derived classes should provide a URL for which help page to use as the 'home' page.
  virtual QString GetHelpHomePageURL() const { return QString(); }

private:

  /// \brief Private method that checks whether or not we are already updating and if not, calls NodeAdded()
  virtual void NodeAddedProxy(const mitk::DataNode *node);

  /// \brief Private method that retrieves the DataStorage from the m_DataStorageServiceTracker
  const mitk::DataStorage* GetDataStorage();

  /// \brief Retrieves the preferences node name, or Null if unsuccessful.
  berry::IPreferences* GetPreferencesNode(const std::string& preferencesNodeName);

  /// \brief Private utility method to calculate min, max, mean and stdDev of an ITK image.
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKGetStatistics(
      itk::Image<TPixel, VImageDimension> *itkImage,
      float &min,
      float &max,
      float &mean,
      float &stdDev);

  ctkPluginContext* m_Context;
  ctkServiceTracker<mitk::IDataStorageService*>* m_DataStorageServiceTracker;
  bool m_InDataStorageChanged;
  static QmitkCommonAppsApplicationPlugin* s_Inst;

};

#endif /* QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_ */
