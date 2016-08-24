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

#include "../niftkBaseApplicationPluginActivator.h"

#include <ctkServiceTracker.h>

#include <itkImage.h>

#include <berryIPreferences.h>
#include <mitkIDataStorageService.h>

#include <niftkLookupTableProviderService.h>


namespace mitk
{
class DataNode;
class DataStorage;
}


namespace niftk
{

const QString IMAGE_INITIALISATION_METHOD_NAME;
const QString IMAGE_INITIALISATION_MIDAS;
const QString IMAGE_INITIALISATION_LEVELWINDOW;
const QString IMAGE_INITIALISATION_PERCENTAGE;
const QString IMAGE_INITIALISATION_PERCENTAGE_NAME;
const QString IMAGE_INITIALISATION_RANGE;
const QString IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME;
const QString IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME;

/**
 * \class PluginActivator
 * \brief Plugin activator for the uk.ac.ucl.cmic.commonapps plugin.
 * \ingroup uk_ac_ucl_cmic_commonapps_internal
 */
class PluginActivator : public BaseApplicationPluginActivator
{
  Q_OBJECT
  Q_INTERFACES(ctkPluginActivator)
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  Q_PLUGIN_METADATA(IID "uk_ac_ucl_cmic_commonapps")
#endif

public:

  PluginActivator();
  virtual ~PluginActivator();

  static PluginActivator* GetInstance();

  virtual void start(ctkPluginContext* context) override;

  virtual void stop(ctkPluginContext* context) override;

protected:

  /**
   * \brief Called when the user toggles the opacity control properties.
   */
  virtual void OnLookupTablePropertyChanged(const itk::Object *caller, const itk::EventObject &event);

  /// \brief Deliberately not virtual method that connects this class to DataStorage so that we can receive NodeAdded events etc.
  void RegisterDataStorageListener();

  /// \brief Deliberately not virtual method that ddisconnects this class from DataStorage so that we can receive NodeAdded events etc.
  void UnregisterDataStorageListener();

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial LevelWindow property to each image.
  void RegisterLevelWindowProperty(const QString& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial "Image Rendering.Mode" property to each image.
  void RegisterImageRenderingModeProperties(const QString& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to register an initial value for Texture Interpolation, and Reslice Interpolation.
  void RegisterInterpolationProperty(const QString& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method that registers initial property values of "outline binary"=true and "opacity"=1 for binary images.
  void RegisterBinaryImageProperties(const QString& preferencesNodeName, mitk::DataNode *constNode);

  /// \brief Deliberately not virtual method thats called by derived classes, to set the departmental logo to blank.
  void BlankDepartmentalLogo();

  /// \brief Called each time a data node is added, and derived classes can override it.
  virtual void NodeAdded(const mitk::DataNode *node);

  /// \brief Called each time a data node is removed, and derived classes can override it.
  virtual void NodeRemoved(const mitk::DataNode *node);

  /// \brief Derived classes should provide a URL for which help page to use as the 'home' page.
  virtual QString GetHelpHomePageURL() const override;

  /// \brief Private method that retrieves the DataStorage from the m_DataStorageServiceTracker
  mitk::DataStorage::Pointer GetDataStorage();

private:

  /// \brief Private method that checks whether or not we are already updating and if not, calls NodeAdded()
  void NodeAddedProxy(const mitk::DataNode *node);

  /// \brief Private method that checks whether or not we are already removing and if not, calls NodeRemoved()
  void NodeRemovedProxy(const mitk::DataNode *node);

  /// \brief Returns the lookup table provider service.
  niftk::LookupTableProviderService* GetLookupTableProvider();

  /// \brief Retrieves the preferences node name, or Null if unsuccessful.
  berry::IPreferences::Pointer GetPreferencesNode(const QString& preferencesNodeName);

  /// \brief Private utility method to calculate min, max, mean and stdDev of an ITK image.
  template<typename TPixel, unsigned int VImageDimension>
  void
  ITKGetStatistics(
      const itk::Image<TPixel, VImageDimension> *itkImage,
      float &min,
      float &max,
      float &mean,
      float &stdDev);

  /// \brief Processes the command line options defined by niftk::BaseApplication.
  void ProcessOptions();

  /// \brief Processes the '--open' command line options.
  void ProcessOpenOptions();

  /// \brief Processes the '--derives-from' command line options.
  void ProcessDerivesFromOptions();

  /// \brief Processes the '--property' command line options.
  void ProcessPropertyOptions();

  /// \brief Parses a node property value that was specified on the command line.
  mitk::BaseProperty::Pointer ParsePropertyValue(const QString& propertyValue);

  ctkServiceTracker<mitk::IDataStorageService*>* m_DataStorageServiceTracker;

  bool m_InDataStorageChanged;

  static PluginActivator* s_Instance;

  std::map<mitk::BaseProperty*, mitk::DataNode*> m_PropertyToNodeMap;
  std::map<mitk::DataNode*, unsigned long int>   m_NodeToLowestOpacityObserverMap;
  std::map<mitk::DataNode*, unsigned long int>   m_NodeToHighestOpacityObserverMap;
  std::map<mitk::DataNode*, unsigned long int>   m_NodeToLookupTableNameObserverMap;

};

}

#endif
