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

#include <mitkCoreServices.h>
#include <mitkIPropertyExtensions.h>
#include <mitkFloatPropertyExtension.h>
#include <mitkProperties.h>
#include <mitkVersion.h>
#include <mitkLogMacros.h>
#include <mitkDataStorage.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkRenderingModeProperty.h>
#include <mitkNamedLookupTableProperty.h>
#include <mitkExceptionMacro.h>
#include <mitkDataNodeFactory.h>
#include <mitkSceneIO.h>
#include <mitkProgressBar.h>

#include <berryPlatformUI.h>

#include <Poco/Util/OptionProcessor.h>

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

#include <QFileInfo>
#include <QDateTime>
#include <QMap>
#include <QtPlugin>
#include <QProcess>
#include <QMainWindow>

#include <NifTKConfigure.h>
#include <mitkDataStorageUtils.h>

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

  std::vector<std::string> args = berry::Platform::GetApplicationArgs();
  QStringList qargs;
  for (std::vector<std::string>::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    qargs << QString::fromStdString(*it);
  }
  // This is a potentially long running operation.
  LoadDataFromDisk(qargs, true);
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
  berry::IPreferencesService::Pointer prefService =
  berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

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
    itk::Image<TPixel, VImageDimension> *itkImage,
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
berry::IPreferences* QmitkCommonAppsApplicationPlugin::GetPreferencesNode(
    const std::string& preferencesNodeName)
{
  berry::IPreferences::Pointer result(NULL);

  berry::IPreferencesService::Pointer prefService =
  berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  if (prefService.IsNotNull())
  {
    result = prefService->GetSystemPreferences()->Node(preferencesNodeName);
  }

  return result.GetPointer();
}

//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::RegisterLevelWindowProperty(
    const std::string& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences* prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode != NULL && image.IsNotNull())
    {
      double percentageOfRange = prefNode->GetDouble(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE_NAME, 50);
      std::string initialisationMethod = prefNode->Get(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_METHOD_NAME, QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS);

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
void QmitkCommonAppsApplicationPlugin::RegisterImageRenderingModeProperties(const std::string& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    berry::IPreferences* prefNode = this->GetPreferencesNode(preferencesNodeName);
    if (prefNode != NULL)
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
    const std::string& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences* prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode != NULL && image.IsNotNull())
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
void QmitkCommonAppsApplicationPlugin::RegisterBinaryImageProperties(const std::string& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences* prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode != NULL && image.IsNotNull())
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
  berry::IPreferencesService::Pointer prefService =
  berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

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

      bool ok = true;
      int argumentsAdded = 0;
      int layer = 0;

      for (int i = 0; i < arguments.size(); ++i)
      {
        if (arguments[i] == "-P"
            || arguments[i] == "--properties")
        {
          ++i;
          if (i == arguments.size())
          {
            MITK_WARN << "Missing command line argument after " << arguments[i - 1].toStdString();
            ok = false;
            break;
          }

          QString propertiesArg = arguments[i];
          QStringList propertiesArgParts = propertiesArg.split(":");
          QString propertyKeysAndValuesPart;
          
          std::vector<mitk::DataNode::Pointer> nodesToSet;
          if (propertiesArgParts.size() == 1)
          {
            nodesToSet = lastOpenedNodes;
            propertyKeysAndValuesPart = propertiesArgParts[0];
          }
          if (propertiesArgParts.size() == 2)
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

          QStringList propertyKeyAndValueParts = propertyKeysAndValuesPart.split(",");
          if (propertyKeyAndValueParts.empty())
          {
            MITK_WARN << "Invalid argument: You must specify properties with the --properties option.";
            ok = false;
            continue;
          }

          foreach (QString propertyKeyAndValuePart, propertyKeyAndValueParts)
          {
            QStringList propertyKeyAndValue = propertyKeyAndValuePart.split("=");
            if (propertyKeyAndValue.size() != 2)
            {
              MITK_WARN << "Invalid argument: You must specify property values in the form <property name>=<value>.";
              ok = false;
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
        else if (arguments[i] == "--open")
        {
          /// Note:
          /// These arguments are processed by the NiftyMIDAS workbench advisor.

          ++i;
          if (i == arguments.size())
          {
            MITK_WARN << "Missing command line argument after " << arguments[i - 1].toStdString();
            ok = false;
            break;
          }

          QString openArg = arguments[i];

          if (openArg.endsWith(".mitk"))
          {
            lastOpenedNodes.clear();

            mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();

            bool clearDataStorageFirst(false);
            mitk::ProgressBar::GetInstance()->AddStepsToDo(2);
            dataStorage = sceneIO->LoadScene( arguments[i].toLocal8Bit().constData(), dataStorage, clearDataStorageFirst );
            mitk::ProgressBar::GetInstance()->Progress(2);
            argumentsAdded++;
          }
          else
          {
            QStringList openArgParts = openArg.split(":");
            QString nodeName;
            QString fileName;
            if (openArgParts.size() == 1)
            {
              fileName = openArgParts[0];
            }
            else if (openArgParts.size() == 2)
            {
              nodeName = openArgParts[0];
              fileName = openArgParts[1];
            }

            mitk::DataNodeFactory::Pointer nodeReader = mitk::DataNodeFactory::New();
            try
            {
              nodeReader->SetFileName(fileName.toStdString());
              nodeReader->Update();

              lastOpenedNodes.clear();

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
                  }
                  
                  ++layer;
                  node->SetIntProperty("layer", layer);
                  node->SetBoolProperty("fixedLayer", true);
                  dataStorage->Add(node);
                  argumentsAdded++;
                }
              }
            }
            catch (...)
            {
              MITK_WARN << "Failed to load command line argument: " << arguments[i].toStdString();
              ok = false;
            }
          }
        }
        else if (arguments[i] == "--perspective"
                 || arguments[i] == "--window-layout"
                 || arguments[i] == "--dnd"
                 || arguments[i] == "--drag-and-drop"
                 || arguments[i] == "--viewer-number"
                 || arguments[i] == "--bind-viewers"
                 )
        {
          /// Note:
          /// These arguments are processed by the NiftyMIDAS workbench advisor.

          ++i;
          if (i == arguments.size())
          {
            MITK_WARN << "Missing command line argument after " << arguments[i - 1].toStdString();
            ok = false;
            break;
          }
        }
        else if (arguments[i].right(5) == ".mitk")
        {
          lastOpenedNodes.clear();

          mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();

          bool clearDataStorageFirst(false);
          mitk::ProgressBar::GetInstance()->AddStepsToDo(2);
          dataStorage = sceneIO->LoadScene( arguments[i].toLocal8Bit().constData(), dataStorage, clearDataStorageFirst );
          mitk::ProgressBar::GetInstance()->Progress(2);
          argumentsAdded++;
        }
        else
        {
          lastOpenedNodes.clear();

          mitk::DataNodeFactory::Pointer nodeReader = mitk::DataNodeFactory::New();
          try
          {
            nodeReader->SetFileName(arguments[i].toStdString());
            nodeReader->Update();
            for (unsigned int j = 0 ; j < nodeReader->GetNumberOfOutputs( ); ++j)
            {
              mitk::DataNode::Pointer node = nodeReader->GetOutput(j);
              if (node->GetData() != 0)
              {
                lastOpenedNodes.push_back(node);

                ++layer;
                node->SetIntProperty("layer", layer);
                node->SetBoolProperty("fixedLayer", true);
                dataStorage->Add(node);
                argumentsAdded++;
              }
            }
          }
          catch(...)
          {
            MITK_WARN << "Failed to load command line argument: " << arguments[i].toStdString();
            ok = false;
          }
        }
      } // end for each command line argument

      if (ok && argumentsAdded > 0 && globalReinit)
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
    propertyTypedValue = QVariant(true);
  }
  else if (propertyValue == QString("false")
           || propertyValue == QString("off")
           || propertyValue == QString("no"))
  {
    propertyTypedValue = QVariant(false);
  }
  else if (propertyValue.size() >= 2
           && ((propertyValue[0] == '\'' && propertyValue[propertyValue.size() - 1] == '\'')
               || (propertyValue[0] == '"' && propertyValue[propertyValue.size() - 1] == '"')))
  {
    propertyTypedValue = QVariant(propertyValue.mid(1, propertyValue.size() - 2));
  }
  else
  {
    bool ok = false;
    int intValue = propertyValue.toInt(&ok);
    if (ok)
    {
      propertyTypedValue = QVariant(intValue);
    }
    else
    {
      double doubleValue = propertyValue.toDouble(&ok);
      if (ok)
      {
        propertyTypedValue = QVariant(doubleValue);
      }
      else
      {
        propertyTypedValue = QVariant(propertyValue);
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
  newArgs << QString("--") + QString::fromStdString(berry::Platform::ARG_NEWINSTANCE);
#else
  newArgs << QString("/") + QString::fromStdString(berry::Platform::ARG_NEWINSTANCE);
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
  berry::IPreferencesService::Pointer prefService
      = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences()->Node("/General");
  bool newInstanceAlways = prefs->GetBool("newInstance.always", false);
  bool newInstanceScene = prefs->GetBool("newInstance.scene", true);

  QStringList args;
  ds >> args;

  QStringList fileArgs;
  QStringList sceneArgs;

  Poco::Util::OptionSet os;
  berry::Platform::GetOptionSet(os);
  Poco::Util::OptionProcessor processor(os);
#if !defined(POCO_OS_FAMILY_UNIX)
  processor.setUnixStyle(false);
#endif
  args.pop_front();
  QStringList::Iterator it = args.begin();
  while (it != args.end())
  {
    std::string name;
    std::string value;
    if (processor.process(it->toStdString(), name, value))
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
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonapps, QmitkCommonAppsApplicationPlugin)
US_INITIALIZE_MODULE("CommonApps", "libuk_ac_ucl_cmic_gui_qt_commonapps")
