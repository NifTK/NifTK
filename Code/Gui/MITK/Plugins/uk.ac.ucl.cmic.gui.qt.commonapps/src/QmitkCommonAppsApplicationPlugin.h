/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_
#define QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_

#include "mitkQtCommonAppsAppDll.h"

#include <berryAbstractUICTKPlugin.h>
#include <ctkServiceTracker.h>
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
class CMIC_QT_COMMONAPPS QmitkCommonAppsApplicationPlugin : public QObject, public berry::AbstractUICTKPlugin
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
  
public:

  QmitkCommonAppsApplicationPlugin();
  ~QmitkCommonAppsApplicationPlugin();

  static QmitkCommonAppsApplicationPlugin* GetDefault();

  ctkPluginContext* GetPluginContext() const;

  virtual void start(ctkPluginContext*);
  virtual void stop(ctkPluginContext*);

  QString GetQtHelpCollectionFile() const;

protected:

  /// \brief Called each time a data node is added, and derived classes can override it.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Derived classes should provide a URL for which help page to use as the 'home' page.
  virtual QString GetHelpHomePageURL() const { return QString(); }

  /// \brief Connects this class to DataStorage so that we can receive NodeAdded events etc.
  virtual void RegisterDataStorageListener();

  /// \brief Disconnects this class from DataStorage so that we can receive NodeAdded events etc.
  virtual void UnregisterDataStorageListener();

  /// \brief Called by derived classes within the start method to set up the help system.
  virtual void RegisterHelpSystem();

  /// \brief Registers the Global Interaction Patterns developed as part of the MIDAS project.
  virtual void RegisterMIDASGlobalInteractionPatterns();

  /// \brief Called by derived classes, to register an initial LevelWindow property to each image.
  virtual void RegisterLevelWindowProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Called by derived classes, to register an initial value for Texture Interpolation, and Reslice Interpolation.
  virtual void RegisterInterpolationProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Called by derived classes, to register an initial value for black opacity property.
  virtual void RegisterBlackOpacityProperty(const std::string& preferencesNodeName, mitk::DataNode *constNode);

  static QmitkCommonAppsApplicationPlugin* inst;
  ctkPluginContext* context;

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

  ctkServiceTracker<mitk::IDataStorageService*>* m_DataStorageServiceTracker;
  bool m_InDataStorageChanged;
};

#endif /* QMITKCOMMONAPPSAPPLICATIONPLUGIN_H_ */
