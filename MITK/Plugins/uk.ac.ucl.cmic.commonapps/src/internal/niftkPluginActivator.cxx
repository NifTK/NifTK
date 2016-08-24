/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <itkCommand.h>
#include <itkStatisticsImageFilter.h>

#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>

#include <berryPlatform.h>
#include <berryPlatformUI.h>
#include <berryIPreferencesService.h>

#include <mitkCoreServices.h>
#include <mitkDataStorage.h>
#include <mitkExceptionMacro.h>
#include <mitkFloatPropertyExtension.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkIOUtil.h>
#include <mitkIPropertyExtensions.h>
#include <mitkLevelWindowProperty.h>
#include <mitkLogMacros.h>
#include <mitkProgressBar.h>
#include <mitkProperties.h>
#include <mitkRenderingModeProperty.h>
#include <mitkVersion.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <mitkRenderingModeProperty.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <usModule.h>
#include <usModuleRegistry.h>
#include <usModuleContext.h>
#include <usModuleInitialization.h>

#include <QApplication>
#include <QDateTime>
#include <QFileInfo>
#include <QMainWindow>
#include <QMap>
#include <QMetaType>
#include <QPair>
#include <QProcess>
#include <QtPlugin>

#include <NifTKConfigure.h>
#include <niftkDataStorageUtils.h>
#include <niftkNamedLookupTableProperty.h>

#include "niftkBaseApplicationPreferencePage.h"
#include "niftkMinimalPerspective.h"


US_INITIALIZE_MODULE


namespace niftk
{

PluginActivator* PluginActivator::s_Instance = nullptr;

//-----------------------------------------------------------------------------
PluginActivator::PluginActivator()
: m_DataStorageServiceTracker(nullptr)
, m_InDataStorageChanged(false)
{
  s_Instance = this;
}


//-----------------------------------------------------------------------------
PluginActivator::~PluginActivator()
{
}


//-----------------------------------------------------------------------------
PluginActivator* PluginActivator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  BaseApplicationPluginActivator::start(context);

  BERRY_REGISTER_EXTENSION_CLASS(MinimalPerspective, context);

  this->RegisterDataStorageListener();
  this->BlankDepartmentalLogo();

  // Get the MitkCore module context.
  us::ModuleContext* mitkCoreContext = us::ModuleRegistry::GetModule(1)->GetModuleContext();

  mitk::IPropertyExtensions* propertyExtensions = mitk::CoreServices::GetPropertyExtensions(mitkCoreContext);
  mitk::FloatPropertyExtension::Pointer opacityPropertyExtension = mitk::FloatPropertyExtension::New(0.0, 1.0);
  propertyExtensions->AddExtension("Image Rendering.Lowest Value Opacity", opacityPropertyExtension.GetPointer());
  propertyExtensions->AddExtension("Image Rendering.Highest Value Opacity", opacityPropertyExtension.GetPointer());

  this->ProcessOptions();
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  this->UnregisterDataStorageListener();

  BaseApplicationPluginActivator::stop(context);
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer PluginActivator::GetDataStorage()
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
void PluginActivator::RegisterDataStorageListener()
{
  m_DataStorageServiceTracker = new ctkServiceTracker<mitk::IDataStorageService*>(this->GetContext());
  m_DataStorageServiceTracker->open();

  this->GetDataStorage()->AddNodeEvent.AddListener
      ( mitk::MessageDelegate1<PluginActivator, const mitk::DataNode*>
        ( this, &PluginActivator::NodeAddedProxy ) );

  this->GetDataStorage()->RemoveNodeEvent.AddListener
      ( mitk::MessageDelegate1<PluginActivator, const mitk::DataNode*>
        ( this, &PluginActivator::NodeRemovedProxy ) );

}


//-----------------------------------------------------------------------------
void PluginActivator::UnregisterDataStorageListener()
{
  if (m_DataStorageServiceTracker != NULL)
  {

    this->GetDataStorage()->AddNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<PluginActivator, const mitk::DataNode*>
          ( this, &PluginActivator::NodeAddedProxy ) );

    this->GetDataStorage()->RemoveNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<PluginActivator, const mitk::DataNode*>
          ( this, &PluginActivator::NodeRemovedProxy ) );

    m_DataStorageServiceTracker->close();
    delete m_DataStorageServiceTracker;
    m_DataStorageServiceTracker = NULL;
  }
}


//-----------------------------------------------------------------------------
void PluginActivator::BlankDepartmentalLogo()
{
  // Blank the departmental logo for now.
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  berry::IPreferences::Pointer logoPref = prefService->GetSystemPreferences()->Node("org.mitk.editors.stdmultiwidget");
  logoPref->Put("DepartmentLogo", "");
}


//-----------------------------------------------------------------------------
QString PluginActivator::GetHelpHomePageURL() const
{
  return QString::null;
}


//-----------------------------------------------------------------------------
void PluginActivator::NodeAddedProxy(const mitk::DataNode *node)
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
void PluginActivator::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);
  this->RegisterInterpolationProperty("uk.ac.ucl.cmic.commonapps", node);
  this->RegisterBinaryImageProperties("uk.ac.ucl.cmic.commonapps", node);
  this->RegisterImageRenderingModeProperties("uk.ac.ucl.cmic.commonapps", node);
  this->RegisterLevelWindowProperty("uk.ac.ucl.cmic.commonapps", node);
}


//-----------------------------------------------------------------------------
void PluginActivator::NodeRemovedProxy(const mitk::DataNode *node)
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
void PluginActivator::NodeRemoved(const mitk::DataNode *constNode)
{
  mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);

  // Removing observers on a node that is being deleted.

  if (niftk::IsNodeAGreyScaleImage(node))
  {
    std::map<mitk::DataNode*, unsigned long int>::iterator it;

    it = m_NodeToLowestOpacityObserverMap.find(node);

    if (it != m_NodeToLowestOpacityObserverMap.end())
    {
      mitk::BaseProperty::Pointer property = node->GetProperty("Image Rendering.Lowest Value Opacity");
      property->RemoveObserver(it->second);
      m_NodeToLowestOpacityObserverMap.erase(it->first);
      m_PropertyToNodeMap.erase(property.GetPointer());
    }

    it = m_NodeToHighestOpacityObserverMap.find(node);

    if (it != m_NodeToHighestOpacityObserverMap.end())
    {
      mitk::BaseProperty::Pointer property = node->GetProperty("Image Rendering.Highest Value Opacity");
      property->RemoveObserver(it->second);

      m_NodeToHighestOpacityObserverMap.erase(it->first);
      m_PropertyToNodeMap.erase(property.GetPointer());
    }

    it = m_NodeToLookupTableNameObserverMap.find(node);

    if (it != m_NodeToLookupTableNameObserverMap.end())
    {
      mitk::BaseProperty::Pointer property = node->GetProperty("LookupTableName");
      property->RemoveObserver(it->second);

      m_NodeToLookupTableNameObserverMap.erase(it->first);
      m_PropertyToNodeMap.erase(property.GetPointer());
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
PluginActivator
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
berry::IPreferences::Pointer PluginActivator::GetPreferencesNode(const QString& preferencesNodeName)
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
void PluginActivator::RegisterLevelWindowProperty(
    const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (niftk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::ConstPointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {
      int minRange = prefNode->GetDouble(IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME, 0);
      int maxRange = prefNode->GetDouble(IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME, 0);
      double percentageOfRange = prefNode->GetDouble(IMAGE_INITIALISATION_PERCENTAGE_NAME, 50);
      QString initialisationMethod = prefNode->Get(IMAGE_INITIALISATION_METHOD_NAME, IMAGE_INITIALISATION_PERCENTAGE);

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
            MITK_ERROR << "Caught exception during PluginActivator::RegisterLevelWindowProperty, so image statistics will be wrong." << e.what();
          }

#if (_MSC_VER == 1700)
          // Visual Studio 2012 does not provide the C++11 std::isnan function.
          if (_isnan(stdDevData))
#else
          if (std::isnan(stdDevData))
#endif
          {
            MITK_WARN << "The image has NaN values. Overriding window/level initialisation mode from MIDAS convention to the mode based on percentage of data range.";
            initialisationMethod = IMAGE_INITIALISATION_PERCENTAGE;
          }

          // This image hasn't had the data members that this view needs (minDataLimit, maxDataLimit etc) initialized yet.
          // i.e. we haven't seen it before. So we have a choice of how to initialise the Level/Window.
          if (initialisationMethod == IMAGE_INITIALISATION_MIDAS)
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
          else if (initialisationMethod == IMAGE_INITIALISATION_PERCENTAGE)
          {
            windowMin = minDataLimit;
            windowMax = minDataLimit + (maxDataLimit - minDataLimit)*percentageOfRange/100.0;
          }
          else if (initialisationMethod == IMAGE_INITIALISATION_RANGE)
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
          MITK_WARN << "PluginActivator::RegisterLevelWindowProperty: Using default Window/Level properties. " << std::endl;
        }

        levelWindow.SetRangeMinMax(minDataLimit, maxDataLimit);
        levelWindow.SetWindowBounds(windowMin, windowMax);
        node->SetLevelWindow(levelWindow);

      } // end if we haven't retrieved the data from the node.
    } // end if have pref node
  } // end if node is grey image
}


//-----------------------------------------------------------------------------
void PluginActivator::OnLookupTablePropertyChanged(const itk::Object* object, const itk::EventObject& /*event*/)
{
  const mitk::BaseProperty* prop = dynamic_cast<const mitk::BaseProperty*>(object);
  assert(prop);

  std::map<mitk::BaseProperty*, mitk::DataNode*>::const_iterator it =
      m_PropertyToNodeMap.find(const_cast<mitk::BaseProperty*>(prop));

  if (it != m_PropertyToNodeMap.end())
  {
    mitk::DataNode *node = it->second;
    if (node != NULL && niftk::IsNodeAGreyScaleImage(node))
    {
      float lowestOpacity = 1;
      bool gotLowest = node->GetFloatProperty("Image Rendering.Lowest Value Opacity", lowestOpacity);

      float highestOpacity = 1;
      bool gotHighest = node->GetFloatProperty("Image Rendering.Highest Value Opacity", highestOpacity);

      std::string defaultName = "grey";
      bool gotIndex = node->GetStringProperty("LookupTableName", defaultName);

      QString lutName = QString::fromStdString(defaultName);

      if (gotLowest && gotHighest && gotIndex)
      {
        // Get LUT from Micro Service.
        niftk::LookupTableProviderService *lutService = this->GetLookupTableProvider();
        niftk::NamedLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(lutName, lowestOpacity, highestOpacity);
        node->SetProperty("LookupTable", mitkLUTProperty);
      }
    }
  }
}


//-----------------------------------------------------------------------------
niftk::LookupTableProviderService* PluginActivator::GetLookupTableProvider()
{
  us::ModuleContext* context = us::GetModuleContext();
  us::ServiceReference<niftk::LookupTableProviderService> ref = context->GetServiceReference<niftk::LookupTableProviderService>();
  niftk::LookupTableProviderService* lutService = context->GetService<niftk::LookupTableProviderService>(ref);

  if (lutService == NULL)
  {
    mitkThrow() << "Failed to find niftk::LookupTableProviderService." << std::endl;
  }

  return lutService;
}


//-----------------------------------------------------------------------------
void PluginActivator::RegisterImageRenderingModeProperties(const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (niftk::IsNodeAGreyScaleImage(node))
  {
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);
    if (prefNode.IsNotNull())
    {
      float lowestOpacity = prefNode->GetFloat(BaseApplicationPreferencePage::LOWEST_VALUE_OPACITY, 1);
      float highestOpacity = prefNode->GetFloat(BaseApplicationPreferencePage::HIGHEST_VALUE_OPACITY, 1);

      mitk::BaseProperty::Pointer lutProp = node->GetProperty("LookupTable");
      const niftk::NamedLookupTableProperty* prop = dynamic_cast<const niftk::NamedLookupTableProperty*>(lutProp.GetPointer());
      if(prop == NULL )
      {
        QString defaultName = "grey";

        // Get LUT from Micro Service.
        niftk::LookupTableProviderService* lutService = this->GetLookupTableProvider();
        niftk::NamedLookupTableProperty::Pointer mitkLUTProperty = lutService->CreateLookupTableProperty(defaultName, lowestOpacity, highestOpacity);

        node->ReplaceProperty("LookupTable", mitkLUTProperty);
        node->SetStringProperty("LookupTableName", defaultName.toStdString().c_str());
        node->SetProperty("Image Rendering.Mode", mitk::RenderingModeProperty::New(mitk::RenderingModeProperty::LOOKUPTABLE_LEVELWINDOW_COLOR));
      }

      node->SetProperty("Image Rendering.Lowest Value Opacity", mitk::FloatProperty::New(lowestOpacity));
      node->SetProperty("Image Rendering.Highest Value Opacity", mitk::FloatProperty::New(highestOpacity));

      if (niftk::IsNodeAGreyScaleImage(node))
      {
        unsigned long int observerId;

        itk::MemberCommand<PluginActivator>::Pointer lookupTablePropertyChangedCommand = itk::MemberCommand<PluginActivator>::New();
        lookupTablePropertyChangedCommand->SetCallbackFunction(this, &PluginActivator::OnLookupTablePropertyChanged);

        mitk::BaseProperty::Pointer lowestIsOpaqueProperty = node->GetProperty("Image Rendering.Lowest Value Opacity");
        observerId = lowestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), lookupTablePropertyChangedCommand);
        m_PropertyToNodeMap.insert(std::make_pair(lowestIsOpaqueProperty.GetPointer(), node));
        m_NodeToLowestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));

        mitk::BaseProperty::Pointer highestIsOpaqueProperty = node->GetProperty("Image Rendering.Highest Value Opacity");
        observerId = highestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), lookupTablePropertyChangedCommand);
        m_PropertyToNodeMap.insert(std::make_pair(highestIsOpaqueProperty.GetPointer(), node));
        m_NodeToHighestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));

        mitk::BaseProperty::Pointer lookupTableNameProperty = node->GetProperty("LookupTableName");
        observerId = lookupTableNameProperty->AddObserver(itk::ModifiedEvent(), lookupTablePropertyChangedCommand);
        m_PropertyToNodeMap.insert(std::make_pair(lookupTableNameProperty.GetPointer(), node));
        m_NodeToLookupTableNameObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));
      }
    } // end if have pref node
  } // end if node is grey image
}


//-----------------------------------------------------------------------------
void PluginActivator::RegisterInterpolationProperty(
    const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (niftk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {

      int imageResliceInterpolation =  prefNode->GetInt(BaseApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION, 2);
      int imageTextureInterpolation =  prefNode->GetInt(BaseApplicationPreferencePage::IMAGE_TEXTURE_INTERPOLATION, 2);

      mitk::BaseProperty::Pointer mitkLUT = node->GetProperty("LookupTable");
      if (mitkLUT.IsNotNull())
      {
        niftk::LabeledLookupTableProperty::Pointer labelProperty
          = dynamic_cast<niftk::LabeledLookupTableProperty*>(mitkLUT.GetPointer());

        if (labelProperty.IsNotNull() && labelProperty->GetIsScaled())
        {
          imageResliceInterpolation = prefNode->GetInt(BaseApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION, 0);
          imageTextureInterpolation = prefNode->GetInt(BaseApplicationPreferencePage::IMAGE_RESLICE_INTERPOLATION, 0);
        }
      }

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
void PluginActivator::RegisterBinaryImageProperties(const QString& preferencesNodeName, mitk::DataNode *node)
{
  if (niftk::IsNodeABinaryImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences::Pointer prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode.IsNotNull() && image.IsNotNull())
    {
      double defaultBinaryOpacity = prefNode->GetDouble(BaseApplicationPreferencePage::BINARY_OPACITY_NAME, BaseApplicationPreferencePage::BINARY_OPACITY_VALUE);
      node->SetOpacity(defaultBinaryOpacity);
      node->SetBoolProperty("outline binary", true);
    } // end if have pref node
  } // end if node is binary image
}


//-----------------------------------------------------------------------------
void PluginActivator::ProcessOptions()
{
  this->ProcessOpenOptions();
  this->ProcessDerivesFromOptions();
  this->ProcessPropertyOptions();
}


//-----------------------------------------------------------------------------
void PluginActivator::ProcessOpenOptions()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  QStringList openArgs = this->GetContext()->getProperty("applicationArgs.open").toStringList();
  int layer = openArgs.size() - 1;

  for (QString openArg: openArgs)
  {
    int colonIndex = openArg.indexOf(':');
    QString nodeNamesPart = openArg.mid(0, colonIndex);
    QString filePath = openArg.mid(colonIndex + 1);

    if (nodeNamesPart.isEmpty())
    {
      MITK_ERROR << "Data node not specified for the '--open' option. Skipping option.";
      continue;
    }

    if (filePath.isEmpty())
    {
      MITK_ERROR << "Data file not specified for the '--open' option. Skipping option.";
      continue;
    }

    if (filePath.right(5) == ".mitk")
    {
      MITK_WARN << "Invalid syntax for opening an MITK project. The '--open' option is for opening single data files with a given name." << std::endl
                << "Omit the '--open' option and provide the file path only." << std::endl;
      continue;
    }

    if (nodeNamesPart.isEmpty())
    {
      MITK_WARN << "Invalid syntax for opening a file. Provide a name for the file. For example:\n"
                   "\n"
                   "    --open T1:/path/to/reference-image.nii.gz\n"
                   "\n"
                   "If you want to use the original name, omit the '--open' option and provide the file path only.\n"
                   "\n";
      continue;
    }

    if (filePath.isEmpty())
    {
      MITK_WARN << "Invalid syntax for opening a file. Provide a path to the file. For example:\n"
                   "\n"
                   "    --open T1:/path/to/reference-image.nii.gz\n"
                   "\n";
      continue;
    }

    try
    {
      for (const QString& nodeName: nodeNamesPart.split(","))
      {
        mitk::DataStorage::SetOfObjects::Pointer nodes = mitk::IOUtil::Load(filePath.toStdString(), *dataStorage);
        int counter = 0;
        for (auto& node: *nodes)
        {
          if (counter == 0)
          {
            node->SetName(nodeName.toStdString().c_str());
          }
          else
          {
            node->SetName(QString("%1 #%2").arg(nodeName, counter + 1).toStdString().c_str());
          }

          node->SetIntProperty("layer", layer--);
        }
      }
    }
    catch (const mitk::Exception& exception)
    {
      MITK_ERROR << "Failed to open file: " << filePath.toStdString();
      MITK_ERROR << exception.what();
    }
  }
}


//-----------------------------------------------------------------------------
void PluginActivator::ProcessDerivesFromOptions()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  /// This set contains the name of the derived nodes from every application of the
  /// --derives-from option. Data node names can occur only once on the right side.
  QSet<QString> everyDerivedNodeName;

  for (QString derivesFromArg: this->GetContext()->getProperty("applicationArgs.derives-from").toStringList())
  {
    int colonIndex = derivesFromArg.indexOf(':');

    QString sourceNodeNamesPart = derivesFromArg.mid(0, colonIndex);
    QString derivedNodeNamesPart = derivesFromArg.mid(colonIndex + 1);

    if (sourceNodeNamesPart.isEmpty())
    {
      MITK_ERROR << "Source data node not specified for the '--derives-from' option. Skipping option.";
      continue;
    }

    if (derivedNodeNamesPart.isEmpty())
    {
      MITK_ERROR << "Data node not specified for the '--derives-from' option. Skipping option.";
      continue;
    }

    bool invalidOption = false;

    mitk::DataStorage::SetOfObjects::Pointer sourceNodes = mitk::DataStorage::SetOfObjects::New();
    for (const QString& sourceNodeName: sourceNodeNamesPart.split(","))
    {
      mitk::DataNode::Pointer sourceNode = dataStorage->GetNamedNode(sourceNodeName.toStdString());

      if (sourceNode.IsNull())
      {
        MITK_ERROR << "Data node not found with the name: " << sourceNodeName.toStdString() << ".\n"
                   << "Make sure you specified a data file with this name or used the '--open' option to open a data file with this name.";
        invalidOption = true;
        break;
      }

      sourceNodes->InsertElement(sourceNodes->Size(), sourceNode);
    }

    if (invalidOption)
    {
      MITK_ERROR << "Skipping option.";
      break;
    }

    for (const QString& derivedNodeName: derivedNodeNamesPart.split(","))
    {
      mitk::DataNode::Pointer derivedNode = dataStorage->GetNamedNode(derivedNodeName.toStdString());

      if (derivedNode.IsNull())
      {
        MITK_ERROR << "Data node not found with the name: " << derivedNodeName.toStdString() << ".\n"
                      "Make sure you specified a data file with this name or used the '--open' option to open a data file with this name.";
        invalidOption = true;
        break;
      }

      if (everyDerivedNodeName.contains(derivedNodeName))
      {
        MITK_ERROR << "Source nodes have already been defined for this data node: " << derivedNodeName.toStdString() << ".";
        invalidOption = true;
        break;
      }

      dataStorage->Remove(derivedNode);
      dataStorage->Add(derivedNode, sourceNodes);
    }

    if (invalidOption)
    {
      MITK_ERROR << "Skipping option.";
      break;
    }
  }
}


//-----------------------------------------------------------------------------
void PluginActivator::ProcessPropertyOptions()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  for (QString propertyArg: this->GetContext()->getProperty("applicationArgs.property").toStringList())
  {
    int colonIndex = propertyArg.indexOf(':');
    QString dataNodeNamesPart = propertyArg.mid(0, colonIndex);
    QString propertyNamesAndValuesPart = propertyArg.mid(colonIndex + 1);

    if (dataNodeNamesPart.isEmpty())
    {
      MITK_ERROR << "Data node not specified for the '--property' option. Skipping option.";
      continue;
    }

    if (propertyNamesAndValuesPart.isEmpty())
    {
      MITK_ERROR << "Property assignments not specified for the '--property' option. Skipping option.";
      continue;
    }

    QStringList dataNodeNames = dataNodeNamesPart.split(",");
    std::vector<mitk::DataNode::Pointer> dataNodes(dataNodeNames.size());
    std::size_t index = 0;

    for (const QString& dataNodeName: dataNodeNames)
    {
      mitk::DataNode::Pointer dataNode = dataStorage->GetNamedNode(dataNodeName.toStdString());

      if (dataNode.IsNull())
      {
        MITK_ERROR << "Data node not found with the name: " << dataNodeName.toStdString() << ".\n"
                      "Make sure you specified a data file with this name or used the '--open' option "
                      "to open a data file with this name.\n"
                      "Skipping setting properties for this data: " << dataNodeName.toStdString() << ".";
        continue;
      }

      dataNodes[index++] = dataNode;
    }

    dataNodes.resize(index);

    for (QString propertyNameAndValuePart: propertyNamesAndValuesPart.split(","))
    {
      QStringList propertyNameAndValue = propertyNameAndValuePart.split("=");
      if (propertyNameAndValue.size() != 2)
      {
        MITK_ERROR << "Invalid argument: You must specify property values in the form <property name>=<value>.";
        continue;
      }

      const QString& propertyName = propertyNameAndValue[0];
      const QString& propertyValue = propertyNameAndValue[1];

      mitk::BaseProperty::Pointer property = this->ParsePropertyValue(propertyValue);

      for (mitk::DataNode::Pointer dataNode: dataNodes)
      {
        /// The level-window property values contain only the lower and upper window
        /// bounds. So that we do not loose other components of the property like the
        /// range min and max values, we retrieve the original level window, and over-
        /// write it with the values specified on the command line.
        /// Note also that we set these properties for the data as well, not only the
        /// node.
        if (mitk::LevelWindowProperty* levelWindowProperty =
            dynamic_cast<mitk::LevelWindowProperty*>(property.GetPointer()))
        {
          mitk::LevelWindow nodeLevelWindow;
          dataNode->GetLevelWindow(nodeLevelWindow, nullptr, propertyName.toStdString().c_str());
          mitk::LevelWindow newLevelWindow = levelWindowProperty->GetLevelWindow();
          nodeLevelWindow.SetWindowBounds(newLevelWindow.GetLowerWindowBound(), newLevelWindow.GetUpperWindowBound());
          levelWindowProperty->SetLevelWindow(nodeLevelWindow);

          dataNode->GetData()->SetProperty(propertyName.toStdString().c_str(), levelWindowProperty);
        }

        dataNode->SetProperty(propertyName.toStdString().c_str(), property);
      }
    }
  }
}


//-----------------------------------------------------------------------------
mitk::BaseProperty::Pointer PluginActivator::ParsePropertyValue(const QString& propertyValue)
{
  mitk::BaseProperty::Pointer property;

  if (propertyValue == QString("true")
      || propertyValue == QString("on")
      || propertyValue == QString("yes"))
  {
    property = mitk::BoolProperty::New(true);
  }
  else if (propertyValue == QString("false")
           || propertyValue == QString("off")
           || propertyValue == QString("no"))
  {
    property = mitk::BoolProperty::New(false);
  }
  else if (propertyValue.size() >= 2
           && ((propertyValue[0] == '\'' && propertyValue[propertyValue.size() - 1] == '\'')
               || (propertyValue[0] == '"' && propertyValue[propertyValue.size() - 1] == '"')))
  {
    property = mitk::StringProperty::New(propertyValue.mid(1, propertyValue.size() - 2).toStdString());
  }
  else
  {
    bool ok = false;
    double doubleValue = propertyValue.toDouble(&ok);
    if (ok)
    {
      if (propertyValue.contains('.') || propertyValue.contains('e') || propertyValue.contains('E'))
      {
        property = mitk::FloatProperty::New(doubleValue);
      }
      else
      {
        int intValue = propertyValue.toInt(&ok);
        if (ok)
        {
          property = mitk::IntProperty::New(intValue);
        }
      }
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
            mitk::LevelWindow levelWindow;
            levelWindow.SetWindowBounds(minValue, maxValue);
            property = mitk::LevelWindowProperty::New(levelWindow);
          }
        }
      }
    }

    if (!ok)
    {
      property = mitk::StringProperty::New(propertyValue.toStdString());
    }
  }

  return property;
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_commonapps, niftk::PluginActivator)
#endif
