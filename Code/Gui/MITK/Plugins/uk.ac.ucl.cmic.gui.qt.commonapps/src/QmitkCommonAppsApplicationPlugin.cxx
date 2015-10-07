/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkCommonAppsApplicationPlugin.h"
#include "QmitkCommonAppsApplicationPreferencePage.h"
#include "QmitkNiftyViewApplicationPreferencePage.h"
#include "internal/QmitkAppInstancesPreferencePage.h"
#include "internal/QmitkModuleView.h"

#include <berryPlatform.h>
#include <berryIPreferencesService.h>

#include <mitkCoreServices.h>
#include <mitkDataNodeFactory.h>
#include <mitkDataStorage.h>
#include <mitkExceptionMacro.h>
#include <mitkFloatPropertyExtension.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkIPropertyExtensions.h>
#include <mitkLevelWindowProperty.h>
#include <mitkLogMacros.h>
#include <mitkNamedLookupTableProperty.h>
#include <mitkProgressBar.h>
#include <mitkProperties.h>
#include <mitkRenderingModeProperty.h>
#include <mitkSceneIO.h>
#include <mitkVersion.h>
#include <mitkVtkResliceInterpolationProperty.h>

#include <berryPlatformUI.h>

#include <itkStatisticsImageFilter.h>
#include <itkCommand.h>
#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <usModule.h>
#include <usModuleRegistry.h>
#include <usModuleContext.h>
#include <usModuleInitialization.h>

#include <QDateTime>
#include <QFileInfo>
#include <QMainWindow>
#include <QMap>
#include <QMetaType>
#include <QPair>
#include <QProcess>
#include <QtPlugin>

#include <NifTKConfigure.h>
#include <mitkDataStorageUtils.h>


US_INITIALIZE_MODULE


/// \brief Helper class to store a pair of double values in a QVariant.
class QLevelWindow : private QPair<double, double>
{
public:
  QLevelWindow()
  {
  }

  void SetWindowBounds(double lowerWindowBound, double upperWindowBound)
  {
    this->first = lowerWindowBound;
    this->second = upperWindowBound;
  }

  void SetLevelWindow(double level, double window)
  {
    this->first = level - window / 2.0;
    this->second = level + window / 2.0;
  }

  double GetLowerWindowBound() const
  {
    return this->first;
  }

  double GetUpperWindowBound() const
  {
    return this->second;
  }

  double GetLevel() const
  {
    return (this->first + this->second) / 2.0;
  }

  double GetWindow() const
  {
    return this->second - this->first;
  }

};

Q_DECLARE_METATYPE(QLevelWindow)


QmitkCommonAppsApplicationPlugin* QmitkCommonAppsApplicationPlugin::s_Inst = 0;

//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin::QmitkCommonAppsApplicationPlugin()
: m_Context(NULL)
, m_DataStorageServiceTracker(NULL)
, m_InDataStorageChanged(false)
{
  s_Inst = this;
}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin::~QmitkCommonAppsApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkCommonAppsApplicationPlugin* QmitkCommonAppsApplicationPlugin::GetDefault()
{
  return s_Inst;
}


//-----------------------------------------------------------------------------
ctkPluginContext* QmitkCommonAppsApplicationPlugin::GetPluginContext() const
{
  return m_Context;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::SetPluginContext(ctkPluginContext* context)
{
  m_Context = context;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::start(ctkPluginContext* context)
{
  this->SetPluginContext(context);

  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsApplicationPreferencePage, m_Context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkAppInstancesPreferencePage, m_Context)
  BERRY_REGISTER_EXTENSION_CLASS(QmitkModuleView, m_Context)

  this->RegisterDataStorageListener();
  this->BlankDepartmentalLogo();

  // Get the MitkCore module context.
  us::ModuleContext* mitkCoreContext = us::ModuleRegistry::GetModule(1)->GetModuleContext();

  mitk::IPropertyExtensions* propertyExtensions = mitk::CoreServices::GetPropertyExtensions(mitkCoreContext);
  mitk::FloatPropertyExtension::Pointer opacityPropertyExtension = mitk::FloatPropertyExtension::New(0.0, 1.0);
  propertyExtensions->AddExtension("Image Rendering.Lowest Value Opacity", opacityPropertyExtension.GetPointer());
  propertyExtensions->AddExtension("Image Rendering.Highest Value Opacity", opacityPropertyExtension.GetPointer());

  /// Note:
  /// Reimplementing functionality from QmitkCommonExtPlugin:

  if (qApp->metaObject()->indexOfSignal("messageReceived(QByteArray)") > -1)
  {
    connect(qApp, SIGNAL(messageReceived(QByteArray)), this, SLOT(handleIPCMessage(QByteArray)));
  }

  QStringList args = berry::Platform::GetApplicationArgs();
  // This is a potentially long running operation.
  LoadDataFromDisk(args, true);
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::stop(ctkPluginContext* context)
{
  this->UnregisterDataStorageListener();
}


//-----------------------------------------------------------------------------
const mitk::DataStorage* QmitkCommonAppsApplicationPlugin::GetDataStorage()
{
  mitk::DataStorage::Pointer dataStorage = NULL;

  if (m_DataStorageServiceTracker != NULL)
  {
    mitk::IDataStorageService* dsService = m_DataStorageServiceTracker->getService();
    if (dsService != 0)
    {
      dataStorage = dsService->GetDataStorage()->GetDataStorage();
    }
  }

  return dataStorage;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterDataStorageListener()
{
  m_DataStorageServiceTracker = new ctkServiceTracker<mitk::IDataStorageService*>(m_Context);
  m_DataStorageServiceTracker->open();

  this->GetDataStorage()->AddNodeEvent.AddListener
      ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
        ( this, &QmitkCommonAppsApplicationPlugin::NodeAddedProxy ) );

  this->GetDataStorage()->RemoveNodeEvent.AddListener
      ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
        ( this, &QmitkCommonAppsApplicationPlugin::NodeRemovedProxy ) );

}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::UnregisterDataStorageListener()
{
  if (m_DataStorageServiceTracker != NULL)
  {

    this->GetDataStorage()->AddNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
          ( this, &QmitkCommonAppsApplicationPlugin::NodeAddedProxy ) );

    this->GetDataStorage()->RemoveNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
          ( this, &QmitkCommonAppsApplicationPlugin::NodeRemovedProxy ) );

    m_DataStorageServiceTracker->close();
    delete m_DataStorageServiceTracker;
    m_DataStorageServiceTracker = NULL;
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterHelpSystem()
{
  ctkServiceReference cmRef = m_Context->getServiceReference<ctkConfigurationAdmin>();
  ctkConfigurationAdmin* configAdmin = 0;
  if (cmRef)
  {
    configAdmin = m_Context->getService<ctkConfigurationAdmin>(cmRef);
  }

  // Use the CTK Configuration Admin service to configure the BlueBerry help system
  if (configAdmin)
  {
    ctkConfigurationPtr conf = configAdmin->getConfiguration("org.blueberry.services.help", QString());
    ctkDictionary helpProps;
    QString urlHomePage = this->GetHelpHomePageURL();
    helpProps.insert("homePage", urlHomePage);
    conf->update(helpProps);
    m_Context->ungetService(cmRef);
  }
  else
  {
    MITK_WARN << "Configuration Admin service unavailable, cannot set home page url.";
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::BlankDepartmentalLogo()
{
  // Blank the departmental logo for now.
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  berry::IPreferences::Pointer logoPref = prefService->GetSystemPreferences()->Node("org.mitk.editors.stdmultiwidget");
  logoPref->Put("DepartmentLogo", "");
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeAddedProxy(const mitk::DataNode *node)
{
  // guarantee no recursions when a new node event is thrown in NodeAdded()
  if (!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);
  this->RegisterInterpolationProperty("uk.ac.ucl.cmic.gui.qt.commonapps", node);
  this->RegisterBinaryImageProperties("uk.ac.ucl.cmic.gui.qt.commonapps", node);
  this->RegisterImageRenderingModeProperties("uk.ac.ucl.cmic.gui.qt.commonapps", node);
  this->RegisterLevelWindowProperty("uk.ac.ucl.cmic.gui.qt.commonapps", node);
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeRemovedProxy(const mitk::DataNode *node)
{
  // guarantee no recursions when a new node event is thrown in NodeRemoved()
  if (!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeRemoved(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::NodeRemoved(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);

  // Removing observers on a node thats being deleted?

  if (mitk::IsNodeAGreyScaleImage(node))
  {
    std::map<mitk::DataNode*, unsigned long int>::iterator lowestIter;
    lowestIter = m_NodeToLowestOpacityObserverMap.find(node);

    std::map<mitk::DataNode*, unsigned long int>::iterator highestIter;
    highestIter = m_NodeToHighestOpacityObserverMap.find(node);

    if (lowestIter != m_NodeToLowestOpacityObserverMap.end())
    {
      if (highestIter != m_NodeToHighestOpacityObserverMap.end())
      {
        mitk::BaseProperty::Pointer lowestIsOpaqueProperty = node->GetProperty("Image Rendering.Lowest Value Opacity");
        lowestIsOpaqueProperty->RemoveObserver(lowestIter->second);

        mitk::BaseProperty::Pointer highestIsOpaqueProperty = node->GetProperty("Image Rendering.Highest Value Opacity");
        highestIsOpaqueProperty->RemoveObserver(highestIter->second);

        m_NodeToLowestOpacityObserverMap.erase(lowestIter->first);
        m_NodeToHighestOpacityObserverMap.erase(highestIter->first);
        m_PropertyToNodeMap.erase(lowestIsOpaqueProperty.GetPointer());
        m_PropertyToNodeMap.erase(highestIsOpaqueProperty.GetPointer());
      }
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
QmitkCommonAppsApplicationPlugin
::ITKGetStatistics(
    const itk::Image<TPixel, VImageDimension> *itkImage,
    float &min,
    float &max,
    float &mean,
    float &stdDev)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::StatisticsImageFilter<ImageType> FilterType;

  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput(itkImage);
  filter->UpdateLargestPossibleRegion();
  min = filter->GetMinimum();
  max = filter->GetMaximum();
  mean = filter->GetMean();
  stdDev = filter->GetSigma();
}


//-----------------------------------------------------------------------------
berry::IPreferences::Pointer QmitkCommonAppsApplicationPlugin::GetPreferencesNode(const QString& preferencesNodeName)
{
  berry::IPreferences::Pointer result(NULL);

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  if (prefService)
  {
    result = prefService->GetSystemPreferences()->Node(preferencesNodeName);
  }

  return result;
}

//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterLevelWindowProperty(
    const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::ConstPointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {
      int minRange = prefNode->GetDouble(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, 0);
      int maxRange = prefNode->GetDouble(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, 0);
      double percentageOfRange = prefNode->GetDouble(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE_NAME, 50);
      QString initialisationMethod = prefNode->Get(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_METHOD_NAME, QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS);

      float minDataLimit(0);
      float maxDataLimit(0);
      float meanData(0);
      float stdDevData(0);

      bool minDataLimitFound = node->GetFloatProperty("image data min", minDataLimit);
      bool maxDataLimitFound = node->GetFloatProperty("image data max", maxDataLimit);
      bool meanDataFound = node->GetFloatProperty("image data mean", meanData);
      bool stdDevDataFound = node->GetFloatProperty("image data std dev", stdDevData);

      if (!minDataLimitFound || !maxDataLimitFound || !meanDataFound || !stdDevDataFound)
      {
        // Provide some defaults.
        minDataLimit = 0;
        maxDataLimit = 255;
        meanData = 15;
        stdDevData = 22;

        // Given that the above values are initial defaults, they must be stored on image.
        node->SetFloatProperty("image data min", minDataLimit);
        node->SetFloatProperty("image data max", maxDataLimit);
        node->SetFloatProperty("image data mean", meanData);
        node->SetFloatProperty("image data std dev", stdDevData);

        // Working data.
        double windowMin = 0;
        double windowMax = 255;
        mitk::LevelWindow levelWindow;

        // We don't have a policy for non-scalar images.
        // For example, how do you default Window/Level for RGB, HSV?
        // So, this stuff below, only valid for scalar images.
        if (image->GetPixelType().GetNumberOfComponents() == 1)
        {
          try
          {
            if (image->GetDimension() == 2)
            {
              AccessFixedDimensionByItk_n(image,
                  ITKGetStatistics, 2,
                  (minDataLimit, maxDataLimit, meanData, stdDevData)
                );
            }
            else if (image->GetDimension() == 3)
            {
              AccessFixedDimensionByItk_n(image,
                  ITKGetStatistics, 3,
                  (minDataLimit, maxDataLimit, meanData, stdDevData)
                );
            }
            else if (image->GetDimension() == 4)
            {
              AccessFixedDimensionByItk_n(image,
                  ITKGetStatistics, 4,
                  (minDataLimit, maxDataLimit, meanData, stdDevData)
                );
            }
            node->SetFloatProperty("image data min", minDataLimit);
            node->SetFloatProperty("image data max", maxDataLimit);
            node->SetFloatProperty("image data mean", meanData);
            node->SetFloatProperty("image data std dev", stdDevData);
            windowMin = minDataLimit;
            windowMax = maxDataLimit;
          }
          catch(const mitk::AccessByItkException& e)
          {
            MITK_ERROR << "Caught exception during QmitkCommonAppsApplicationPlugin::RegisterLevelWindowProperty, so image statistics will be wrong." << e.what();
          }

          // This image hasn't had the data members that this view needs (minDataLimit, maxDataLimit etc) initialized yet.
          // i.e. we haven't seen it before. So we have a choice of how to initialise the Level/Window.
          if (initialisationMethod == QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS)
          {
            double centre = (minDataLimit + 4.51*stdDevData)/2.0;
            double width = 4.5*stdDevData;
            windowMin = centre - width/2.0;
            windowMax = centre + width/2.0;

            if (windowMin < minDataLimit)
            {
              windowMin = minDataLimit;
            }
            if (windowMax > maxDataLimit)
            {
              windowMax = maxDataLimit;
            }
          }
          else if (initialisationMethod == QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE)
          {
            windowMin = minDataLimit;
            windowMax = minDataLimit + (maxDataLimit - minDataLimit)*percentageOfRange/100.0;
          }
          else if (initialisationMethod == QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_RANGE)
          {
            windowMin = minRange; // ignores data completely.
            windowMax = maxRange; // ignores data completely.
          }
          else
          {
            // Do nothing, which means the MITK framework will pick one.
          }
        }
        else
        {
          MITK_WARN << "QmitkCommonAppsApplicationPlugin::RegisterLevelWindowProperty: Using default Window/Level properties. " << std::endl;
        }

        levelWindow.SetRangeMinMax(minDataLimit, maxDataLimit);
        levelWindow.SetWindowBounds(windowMin, windowMax);
        node->SetLevelWindow(levelWindow);

      } // end if we haven't retrieved the data from the node.
    } // end if have pref node
  } // end if node is grey image
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::OnLookupTablePropertyChanged(const itk::Object *object, const itk::EventObject & event)
{
  const mitk::BaseProperty* prop = dynamic_cast<const mitk::BaseProperty*>(object);
  if (prop != NULL)
  {
    std::map<mitk::BaseProperty*, mitk::DataNode*>::const_iterator iter;
    iter = m_PropertyToNodeMap.find(const_cast<mitk::BaseProperty*>(prop));
    if (iter != m_PropertyToNodeMap.end())
    {
      mitk::DataNode *node = iter->second;
      if (node != NULL && mitk::IsNodeAGreyScaleImage(node))
      {
        float lowestOpacity = 1;
        bool gotLowest = node->GetFloatProperty("Image Rendering.Lowest Value Opacity", lowestOpacity);

        float highestOpacity = 1;
        bool gotHighest = node->GetFloatProperty("Image Rendering.Highest Value Opacity", highestOpacity);

        int lookupTableIndex = 0;
        bool gotIndex = node->GetIntProperty("LookupTableIndex", lookupTableIndex);

        if (gotLowest && gotHighest && gotIndex)
        {
          // Get LUT from Micro Service.
          QmitkLookupTableProviderService *lutService = this->GetLookupTableProvider();
          mitk::NamedLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(lookupTableIndex, lowestOpacity, highestOpacity);
          node->SetProperty("LookupTable", mitkLUTProperty);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
QmitkLookupTableProviderService* QmitkCommonAppsApplicationPlugin::GetLookupTableProvider()
{
  us::ModuleContext* context = us::GetModuleContext();
  us::ServiceReference<QmitkLookupTableProviderService> ref = context->GetServiceReference<QmitkLookupTableProviderService>();
  QmitkLookupTableProviderService* lutService = context->GetService<QmitkLookupTableProviderService>(ref);

  if (lutService == NULL)
  {
    mitkThrow() << "Failed to find QmitkLookupTableProviderService." << std::endl;
  }

  return lutService;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterImageRenderingModeProperties(const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);
    if (prefNode.IsNotNull())
    {
      float lowestOpacity = prefNode->GetFloat(QmitkCommonAppsApplicationPreferencePage::LOWEST_VALUE_OPACITY, 1);
      float highestOpacity = prefNode->GetFloat(QmitkCommonAppsApplicationPreferencePage::HIGHEST_VALUE_OPACITY, 1);
      unsigned int defaultIndex = 0;

      // Get LUT from Micro Service.
      QmitkLookupTableProviderService *lutService = this->GetLookupTableProvider();
      mitk::NamedLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(defaultIndex, lowestOpacity, highestOpacity);

      node->ReplaceProperty("LookupTable", mitkLUTProperty);
      node->SetIntProperty("LookupTableIndex", defaultIndex);
      node->SetProperty("Image Rendering.Mode", mitk::RenderingModeProperty::New(mitk::RenderingModeProperty::LOOKUPTABLE_LEVELWINDOW_COLOR));
      node->SetProperty("Image Rendering.Lowest Value Opacity", mitk::FloatProperty::New(lowestOpacity));
      node->SetProperty("Image Rendering.Highest Value Opacity", mitk::FloatProperty::New(highestOpacity));

      if (mitk::IsNodeAGreyScaleImage(node))
      {
        unsigned long int observerId;

        itk::MemberCommand<QmitkCommonAppsApplicationPlugin>::Pointer lowestIsOpaqueCommand = itk::MemberCommand<QmitkCommonAppsApplicationPlugin>::New();
        lowestIsOpaqueCommand->SetCallbackFunction(this, &QmitkCommonAppsApplicationPlugin::OnLookupTablePropertyChanged);
        mitk::BaseProperty::Pointer lowestIsOpaqueProperty = node->GetProperty("Image Rendering.Lowest Value Opacity");
        observerId = lowestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), lowestIsOpaqueCommand);
        m_PropertyToNodeMap.insert(std::pair<mitk::BaseProperty*, mitk::DataNode*>(lowestIsOpaqueProperty.GetPointer(), node));
        m_NodeToLowestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));

        itk::MemberCommand<QmitkCommonAppsApplicationPlugin>::Pointer highestIsOpaqueCommand = itk::MemberCommand<QmitkCommonAppsApplicationPlugin>::New();
        highestIsOpaqueCommand->SetCallbackFunction(this, &QmitkCommonAppsApplicationPlugin::OnLookupTablePropertyChanged);
        mitk::BaseProperty::Pointer highestIsOpaqueProperty = node->GetProperty("Image Rendering.Highest Value Opacity");
        observerId = highestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), highestIsOpaqueCommand);
        m_PropertyToNodeMap.insert(std::pair<mitk::BaseProperty*, mitk::DataNode*>(highestIsOpaqueProperty.GetPointer(), node));
        m_NodeToHighestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));
      }
    } // end if have pref node
  } // end if node is grey image
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterInterpolationProperty(
    const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {

      int imageResliceInterpolation =  prefNode->GetInt(QmitkCommonAppsApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION, 2);
      int imageTextureInterpolation =  prefNode->GetInt(QmitkCommonAppsApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION, 2);

      if (imageTextureInterpolation == 0)
      {
        node->SetProperty("texture interpolation", mitk::BoolProperty::New(false));
      }
      else
      {
        node->SetProperty("texture interpolation", mitk::BoolProperty::New(true));
      }

      mitk::VtkResliceInterpolationProperty::Pointer interpolationProperty = mitk::VtkResliceInterpolationProperty::New();

      if (imageResliceInterpolation == 0)
      {
        interpolationProperty->SetInterpolationToNearest();
      }
      else if (imageResliceInterpolation == 1)
      {
        interpolationProperty->SetInterpolationToLinear();
      }
      else if (imageResliceInterpolation == 2)
      {
        interpolationProperty->SetInterpolationToCubic();
      }
      node->SetProperty("reslice interpolation", interpolationProperty);

    } // end if have pref node
  } // end if node is grey image
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterBinaryImageProperties(const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {
      double defaultBinaryOpacity = prefNode->GetDouble(QmitkCommonAppsApplicationPreferencePage::BINARY_OPACITY_NAME, QmitkCommonAppsApplicationPreferencePage::BINARY_OPACITY_VALUE);
      node->SetOpacity(defaultBinaryOpacity);
      node->SetBoolProperty("outline binary", true);
    } // end if have pref node
  } // end if node is binary image
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::SetFileOpenTriggersReinit(bool openEditor)
{
  // Blank the departmental logo for now.
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  berry::IPreferences::Pointer generalPrefs = prefService->GetSystemPreferences()->Node("/General");
  generalPrefs->PutBool("OpenEditor", openEditor);
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::LoadDataFromDisk(const QStringList &arguments, bool globalReinit)
{
  if (!arguments.empty())
  {
    ctkServiceReference serviceRef = m_Context->getServiceReference<mitk::IDataStorageService>();
    if (serviceRef)
    {
      mitk::IDataStorageService* dataStorageService = m_Context->getService<mitk::IDataStorageService>(serviceRef);
      mitk::DataStorage::Pointer dataStorage = dataStorageService->GetDefaultDataStorage()->GetDataStorage();

      std::vector<mitk::DataNode::Pointer> lastOpenedNodes;

      int filesOpened = 0;
      int layer = 0;

      for (int i = 0; i < arguments.size(); ++i)
      {
        if (arguments[i] == "-o"
            || arguments[i] == "--open")
        {
          /// --open should be followed by a file path, not an option.
          if (i + 1 == arguments.size()
              || arguments[i + 1].isEmpty()
              || arguments[i + 1][0] == '-')
          {
            MITK_ERROR << "Missing command line argument after the " << arguments[i].toStdString() << " option.";
            continue;
          }
        }
        else if (arguments[i] == "-p"
                 || arguments[i] == "--parents")
        {
          /// --parents should be followed by data node names, not by an option.
          if (i + 1 == arguments.size()
              || arguments[i + 1].isEmpty()
              || arguments[i + 1][0] == '-')
          {
            MITK_ERROR << "Missing command line argument after the " << arguments[i].toStdString() << " option.";
            continue;
          }

          ++i;
          QString sourcesArg = arguments[i];

          QStringList sourceNodeNames = sourcesArg.split(",");
          if (sourceNodeNames.empty())
          {
            MITK_ERROR << "Invalid argument: You must specify data names with the " << arguments[i - 1].toStdString() << " option.";
            continue;
          }

          mitk::DataStorage::SetOfObjects::Pointer lastOpenedNodesSources = mitk::DataStorage::SetOfObjects::New();
          foreach (QString sourceNodeName, sourceNodeNames)
          {
            mitk::DataNode::Pointer sourceNode = dataStorage->GetNamedNode(sourceNodeName.toStdString());
            if (sourceNode.IsNull())
            {
              MITK_ERROR << "The name has not be given to any data: " << sourceNodeName.toStdString();
              continue;
            }
            lastOpenedNodesSources->push_back(sourceNode);
          }

          for (std::vector<mitk::DataNode::Pointer>::iterator lastOpenedNodesIt = lastOpenedNodes.begin();
               lastOpenedNodesIt != lastOpenedNodes.end();
               ++lastOpenedNodesIt)
          {
            if (dataStorage->Exists(*lastOpenedNodesIt))
            {
              dataStorage->Remove(*lastOpenedNodesIt);
            }
            dataStorage->Add(*lastOpenedNodesIt, lastOpenedNodesSources);
          }
        }
        else if (arguments[i] == "-P"
                 || arguments[i] == "--properties")
        {
          /// --properties should be followed by property values, not an option.
          if (i + 1 == arguments.size()
              || arguments[i + 1].isEmpty()
              || arguments[i + 1][0] == '-')
          {
            MITK_ERROR << "Missing command line argument after the " << arguments[i].toStdString() << " option.";
            continue;
          }

          ++i;
          QString propertiesArg = arguments[i];
          QStringList propertiesArgParts = propertiesArg.split(":");
          QString propertyKeysAndValuesPart;
          
          std::vector<mitk::DataNode::Pointer> nodesToSet;
          if (propertiesArgParts.size() == 1)
          {
            nodesToSet = lastOpenedNodes;
            propertyKeysAndValuesPart = propertiesArgParts[0];
          }
          else if (propertiesArgParts.size() == 2)
          {
            QString nodeNamesPart = propertiesArgParts[0];
            QStringList nodeNames = nodeNamesPart.split(",");
            foreach (QString nodeName, nodeNames)
            {
              mitk::DataNode::Pointer node = dataStorage->GetNamedNode(nodeName.toStdString());
              if (node.IsNull())
              {
                MITK_ERROR << "The name has not be given to any data: " << nodeName.toStdString();
                continue;
              }
              nodesToSet.push_back(node);
            }

            propertyKeysAndValuesPart = propertiesArgParts[1];
          }
          else
          {
            MITK_ERROR << "Invalid syntax for the " << arguments[i - 1].toStdString() << " option.";
            continue;
          }

          QStringList propertyKeyAndValueParts = propertyKeysAndValuesPart.split(",");
          if (propertyKeyAndValueParts.empty())
          {
            MITK_ERROR << "Invalid argument: You must specify properties with the " << arguments[i - 1].toStdString() << " option.";
            continue;
          }

          foreach (QString propertyKeyAndValuePart, propertyKeyAndValueParts)
          {
            QStringList propertyKeyAndValue = propertyKeyAndValuePart.split("=");
            if (propertyKeyAndValue.size() != 2)
            {
              MITK_ERROR << "Invalid argument: You must specify property values in the form <property name>=<value>.";
              continue;
            }

            QString propertyKey = propertyKeyAndValue[0];
            QString propertyValue = propertyKeyAndValue[1];

            for (std::vector<mitk::DataNode::Pointer>::iterator nodesIt = nodesToSet.begin();
                 nodesIt != nodesToSet.end();
                 ++nodesIt)
            {
              QVariant value = this->ParsePropertyValue(propertyValue);
              this->SetNodeProperty(*nodesIt, propertyKey, value);
            }
          }
        }
        else if (arguments[i] == "--perspective"
                 || arguments[i] == "--window-layout"
                 || arguments[i] == "--dnd"
                 || arguments[i] == "--drag-and-drop"
                 || arguments[i] == "--viewer-number"
                 || arguments[i] == "--bind-viewers"
                 || arguments[i] == "--bind-windows"
                 )
        {
          /// Note:
          /// These arguments are processed by the NiftyMIDAS workbench advisor.

          if (i + 1 == arguments.size()
              || arguments[i + 1].isEmpty()
              || arguments[i + 1][0] == '-')
          {
            MITK_ERROR << "Missing command line argument after the " << arguments[i].toStdString() << " option.";
            continue;
          }

          ++i;
        }
        else if (arguments[i].right(5) == ".mitk")
        {
          lastOpenedNodes.clear();

          mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();

          bool clearDataStorageFirst(false);
          mitk::ProgressBar::GetInstance()->AddStepsToDo(2);
          dataStorage = sceneIO->LoadScene( arguments[i].toLocal8Bit().constData(), dataStorage, clearDataStorageFirst );
          mitk::ProgressBar::GetInstance()->Progress(2);
          ++filesOpened;
        }
        else
        {
          QString fileArg = arguments[i];

          QStringList fileArgParts = fileArg.split(":");
          QString nodeName;
          QString filePath;
          if (fileArgParts.size() == 1)
          {
            filePath = fileArgParts[0];
          }
          else if (fileArgParts.size() == 2)
          {
            nodeName = fileArgParts[0];
            filePath = fileArgParts[1];
          }
          else
          {
            MITK_ERROR << "Invalid syntax for specifying input file.";
            break;
          }

          lastOpenedNodes.clear();

          /// If parents are specified for these nodes then the --parents option should follow
          /// this argument either directly or after the properties option.
          bool parentsSpecified;
          if ((i + 1 < arguments.size()
               && (arguments[i + 1] == "-p" || arguments[i + 1] == "--parents"))
              || (i + 3 < arguments.size()
                  && (arguments[i + 1] == "-P" || arguments[i + 1] == "--properties")
                  && (arguments[i + 3] == "-p" || arguments[i + 3] == "--parents")))
          {
            parentsSpecified = true;
          }
          else
          {
            parentsSpecified = false;
          }

          mitk::DataNodeFactory::Pointer nodeReader = mitk::DataNodeFactory::New();
          try
          {
            nodeReader->SetFileName(filePath.toStdString());
            nodeReader->Update();
            for (unsigned int j = 0 ; j < nodeReader->GetNumberOfOutputs(); ++j)
            {
              mitk::DataNode::Pointer node = nodeReader->GetOutput(j);
              if (node->GetData() != 0)
              {
                lastOpenedNodes.push_back(node);

                if (!nodeName.isEmpty())
                {
                  if (j == 0)
                  {
                    node->SetName(nodeName.toStdString().c_str());
                  }
                  else
                  {
                    node->SetName(QString("%1 #%2").arg(nodeName, j + 1).toStdString().c_str());
                  }
                  ++filesOpened;
                }

                ++layer;
                node->SetIntProperty("layer", layer);
                node->SetBoolProperty("fixedLayer", true);

                if (!parentsSpecified)
                {
                  dataStorage->Add(node);
                }
              }
            }
          }
          catch (...)
          {
            MITK_ERROR << "Failed to open file: " << filePath.toStdString();
          }
        }
      } // end for each command line argument

      if (filesOpened > 0 && globalReinit)
      {
        // calculate bounding geometry
        mitk::RenderingManager::GetInstance()->InitializeViews(dataStorage->ComputeBoundingGeometry3D());
      }
    }
    else
    {
      MITK_ERROR << "A service reference for mitk::IDataStorageService does not exist";
    }
  }
}


//-----------------------------------------------------------------------------
QVariant QmitkCommonAppsApplicationPlugin::ParsePropertyValue(const QString& propertyValue)
{
  QVariant propertyTypedValue;

  if (propertyValue == QString("true")
      || propertyValue == QString("on")
      || propertyValue == QString("yes"))
  {
    propertyTypedValue.setValue(true);
  }
  else if (propertyValue == QString("false")
           || propertyValue == QString("off")
           || propertyValue == QString("no"))
  {
    propertyTypedValue.setValue(false);
  }
  else if (propertyValue.size() >= 2
           && ((propertyValue[0] == '\'' && propertyValue[propertyValue.size() - 1] == '\'')
               || (propertyValue[0] == '"' && propertyValue[propertyValue.size() - 1] == '"')))
  {
    propertyTypedValue.setValue(propertyValue.mid(1, propertyValue.size() - 2));
  }
  else
  {
    bool ok = false;
    int intValue = propertyValue.toInt(&ok);
    if (ok)
    {
      propertyTypedValue.setValue(intValue);
    }
    else
    {
      double doubleValue = propertyValue.toDouble(&ok);
      if (ok)
      {
        propertyTypedValue.setValue(doubleValue);
      }
      else
      {
        int hyphenIndex = propertyValue.indexOf('-', 1);
        if (hyphenIndex != -1)
        {
          /// It might be a level window min-max range.
          QString minPart = propertyValue.mid(0, hyphenIndex);
          QString maxPart = propertyValue.mid(hyphenIndex + 1, propertyValue.length() - hyphenIndex);
          double minValue = minPart.toDouble(&ok);
          if (ok)
          {
            double maxValue = maxPart.toDouble(&ok);
            if (ok)
            {
              QLevelWindow range;
              range.SetWindowBounds(minValue, maxValue);
              propertyTypedValue.setValue(range);
            }
          }
        }

        if (!ok)
        {
          propertyTypedValue.setValue(propertyValue);
        }
      }
    }
  }

  return propertyTypedValue;
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::SetNodeProperty(mitk::DataNode* node, const QString& propertyName, const QVariant& propertyValue, const QString& rendererName)
{
  mitk::BaseProperty::Pointer mitkProperty;
  if (propertyValue.type() == QVariant::Bool)
  {
    mitkProperty = mitk::BoolProperty::New(propertyValue.toBool());
  }
  else if (propertyValue.type() == QVariant::Int)
  {
    mitkProperty = mitk::IntProperty::New(propertyValue.toInt());
  }
  else if (propertyValue.type() == QVariant::Double)
  {
    mitkProperty = mitk::FloatProperty::New(propertyValue.toFloat());
  }
  else if (propertyValue.type() == QVariant::String)
  {
    mitkProperty = mitk::StringProperty::New(propertyValue.toString().toStdString());
  }
  else if (propertyValue.type() == QVariant::UserType)
  {
    if (propertyValue.canConvert<QLevelWindow>())
    {
      QLevelWindow qLevelWindow = propertyValue.value<QLevelWindow>();
      mitk::LevelWindow levelWindow;
      node->GetLevelWindow(levelWindow);
      levelWindow.SetWindowBounds(qLevelWindow.GetLowerWindowBound(), qLevelWindow.GetUpperWindowBound());
      node->SetLevelWindow(levelWindow);
      node->GetData()->SetProperty("levelwindow", mitk::LevelWindowProperty::New(levelWindow));
    }
  }

  if (rendererName.isEmpty())
  {
    node->SetProperty(propertyName.toStdString().c_str(), mitkProperty);
  }
  else
  {
    node->GetPropertyList(rendererName.toStdString())->SetProperty(propertyName.toStdString().c_str(), mitkProperty);
  }
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::startNewInstance(const QStringList &args, const QStringList& files)
{
  QStringList newArgs(args);
#ifdef Q_OS_UNIX
  newArgs << QString("--") + berry::Platform::PROP_NEWINSTANCE;
#else
  newArgs << QString("/") + berry::Platform::PROP_NEWINSTANCE;
#endif
  newArgs << files;
  QProcess::startDetached(qApp->applicationFilePath(), newArgs);
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::handleIPCMessage(const QByteArray& msg)
{
  QDataStream ds(msg);
  QString msgType;
  ds >> msgType;

  // we only handle messages containing command line arguments
  if (msgType != "$cmdLineArgs") return;

  // activate the current workbench window
  berry::IWorkbenchWindow::Pointer window =
      berry::PlatformUI::GetWorkbench()->GetActiveWorkbenchWindow();

  QMainWindow* mainWindow =
   static_cast<QMainWindow*> (window->GetShell()->GetControl());

  mainWindow->setWindowState(mainWindow->windowState() & ~Qt::WindowMinimized);
  mainWindow->raise();
  mainWindow->activateWindow();

  // Get the preferences for the instantiation behavior
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences()->Node("/General");
  bool newInstanceAlways = prefs->GetBool("newInstance.always", false);
  bool newInstanceScene = prefs->GetBool("newInstance.scene", true);

  QStringList args;
  ds >> args;

  QStringList fileArgs;
  QStringList sceneArgs;

  QStringList applicationArgs = berry::Platform::GetApplicationArgs();
  args.pop_front();
  QStringList::Iterator it = args.begin();
  while (it != args.end())
  {
    if (it->startsWith("-"))
    {
      ++it;
    }
    else
    {
      if (it->endsWith(".mitk"))
      {
        sceneArgs << *it;
      }
      else
      {
        fileArgs << *it;
      }
      it = args.erase(it);
    }
  }

  if (newInstanceAlways)
  {
    if (newInstanceScene)
    {
      startNewInstance(args, fileArgs);

      foreach(QString sceneFile, sceneArgs)
      {
        startNewInstance(args, QStringList(sceneFile));
      }
    }
    else
    {
      fileArgs.append(sceneArgs);
      startNewInstance(args, fileArgs);
    }
  }
  else
  {
    LoadDataFromDisk(fileArgs, false);
    if (newInstanceScene)
    {
      foreach(QString sceneFile, sceneArgs)
      {
        startNewInstance(args, QStringList(sceneFile));
      }
    }
    else
    {
      LoadDataFromDisk(sceneArgs, false);
    }
  }

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonapps, QmitkCommonAppsApplicationPlugin)
#endif
