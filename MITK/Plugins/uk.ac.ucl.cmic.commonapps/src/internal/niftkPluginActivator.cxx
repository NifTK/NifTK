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
#include <mitkDataNodeFactory.h>
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
#include <mitkSceneIO.h>
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

  // Removing observers on a node thats being deleted?

  if (niftk::IsNodeAGreyScaleImage(node))
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
void PluginActivator::OnLookupTablePropertyChanged(const itk::Object *object, const itk::EventObject & event)
{
  const mitk::BaseProperty* prop = dynamic_cast<const mitk::BaseProperty*>(object);
  if (prop != NULL)
  {
    std::map<mitk::BaseProperty*, mitk::DataNode*>::const_iterator iter;
    iter = m_PropertyToNodeMap.find(const_cast<mitk::BaseProperty*>(prop));
    if (iter != m_PropertyToNodeMap.end())
    {
      mitk::DataNode *node = iter->second;
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

        itk::MemberCommand<PluginActivator>::Pointer lowestIsOpaqueCommand = itk::MemberCommand<PluginActivator>::New();
        lowestIsOpaqueCommand->SetCallbackFunction(this, &PluginActivator::OnLookupTablePropertyChanged);
        mitk::BaseProperty::Pointer lowestIsOpaqueProperty = node->GetProperty("Image Rendering.Lowest Value Opacity");
        observerId = lowestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), lowestIsOpaqueCommand);
        m_PropertyToNodeMap.insert(std::pair<mitk::BaseProperty*, mitk::DataNode*>(lowestIsOpaqueProperty.GetPointer(), node));
        m_NodeToLowestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));

        itk::MemberCommand<PluginActivator>::Pointer highestIsOpaqueCommand = itk::MemberCommand<PluginActivator>::New();
        highestIsOpaqueCommand->SetCallbackFunction(this, &PluginActivator::OnLookupTablePropertyChanged);
        mitk::BaseProperty::Pointer highestIsOpaqueProperty = node->GetProperty("Image Rendering.Highest Value Opacity");
        observerId = highestIsOpaqueProperty->AddObserver(itk::ModifiedEvent(), highestIsOpaqueCommand);
        m_PropertyToNodeMap.insert(std::pair<mitk::BaseProperty*, mitk::DataNode*>(highestIsOpaqueProperty.GetPointer(), node));
        m_NodeToHighestOpacityObserverMap.insert(std::pair<mitk::DataNode*, unsigned long int>(node, observerId));
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
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  QStringList openArgs = this->GetContext()->getProperty("applicationArgs.open").toStringList();
  int layer = openArgs.size() - 1;

  for (QString openArg: openArgs)
  {
    int colonIndex = openArg.indexOf(':');
    QString nodeName = openArg.mid(0, colonIndex);
    QString filePath = openArg.mid(colonIndex + 1);

    if (filePath.right(5) == ".mitk")
    {
      MITK_WARN << "Invalid syntax for opening an MITK project. The '--open' option is for opening single data files with a given name." << std::endl
                << "Omit the '--open' option and provide the file path only." << std::endl;
      continue;
    }

    if (nodeName.isEmpty())
    {
      MITK_WARN << "Invalid syntax for opening a file. Provide a name for the file. For example:" << std::endl
                << std::endl
                << "    --open T1:/path/to/reference-image.nii.gz" << std::endl
                << std::endl
                << "If you want to use the original name, omit the '--open' option and provide the file path only." << std::endl
                << std::endl;
      continue;
    }

    if (filePath.isEmpty())
    {
      MITK_WARN << "Invalid syntax for opening a file. Provide a path to the file. For example:" << std::endl
                << std::endl
                << "    --open T1:/path/to/reference-image.nii.gz" << std::endl
                << std::endl;
      continue;
    }

    try
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
    catch (const mitk::Exception& exception)
    {
      MITK_ERROR << "Failed to open file: " << filePath.toStdString();
      MITK_ERROR << exception.what();
    }
  }

  std::map<mitk::DataNode::Pointer, mitk::DataStorage::SetOfObjects::Pointer> sourcesOfDerivedNodes;

  for (QString derivesFromArg: this->GetContext()->getProperty("applicationArgs.derives-from").toStringList())
  {
    int colonIndex = derivesFromArg.indexOf(':');
    QString sourceDataName = derivesFromArg.mid(0, colonIndex);
    QString derivedDataNamesPart = derivesFromArg.mid(colonIndex + 1);
    QStringList derivedDataNames = derivedDataNamesPart.split(",");

    mitk::DataNode::Pointer sourceNode = dataStorage->GetNamedNode(sourceDataName.toStdString());

    if (sourceNode.IsNull())
    {
      MITK_ERROR << "Data not found with the name: " << sourceDataName.toStdString() << std::endl
                 << "Make sure you specified a data file with this name or used the '--open' option "
                    "to open a data file with this name." << std::endl
                 << "Skipping adding these as derived data for this source: " << derivedDataNamesPart.toStdString() << ".";
      continue;
    }

    for (const QString& derivedDataName: derivedDataNames)
    {
      mitk::DataNode::Pointer derivedNode = dataStorage->GetNamedNode(derivedDataName.toStdString());

      if (derivedNode.IsNull())
      {
        MITK_ERROR << "Data not found with the name: " << derivedDataName.toStdString() << std::endl
                   << "Make sure you specified a data file with this name or used the '--open' option"
                      "to open a data file with this name." << std::endl
                   << "Skipping adding this data as derived data for the source: " << sourceDataName.toStdString() << ".";
        continue;
      }

      auto sourcesOfDerivedNodesIt = sourcesOfDerivedNodes.find(sourceNode);
      if (sourcesOfDerivedNodesIt == sourcesOfDerivedNodes.end())
      {
        mitk::DataStorage::SetOfObjects::Pointer sourceDataNodes = mitk::DataStorage::SetOfObjects::New();
        sourcesOfDerivedNodesIt = sourcesOfDerivedNodes.insert(std::make_pair(derivedNode, sourceDataNodes)).first;
      }
      sourcesOfDerivedNodesIt->second->InsertElement(0, sourceNode);
    }
  }

  for (const auto& sourcesOfDerivedNode: sourcesOfDerivedNodes)
  {
    const auto& derivedNode = sourcesOfDerivedNode.first;
    const auto& sourceNodes = sourcesOfDerivedNode.second;
    dataStorage->Remove(derivedNode);
    dataStorage->Add(derivedNode, sourceNodes);
  }

  for (QString propertyArg: this->GetContext()->getProperty("applicationArgs.property").toStringList())
  {
    int colonIndex = propertyArg.indexOf(':');
    QString dataNodeName = propertyArg.mid(0, colonIndex);
    QString propertyNamesAndValuesPart = propertyArg.mid(colonIndex + 1);

    mitk::DataNode::Pointer dataNode = dataStorage->GetNamedNode(dataNodeName.toStdString());

    if (dataNode.IsNull())
    {
      MITK_ERROR << "Data not found with the name: " << dataNodeName.toStdString() << std::endl
                 << "Make sure you specified a data file with this name or used the '--open' option "
                    "to open a data file with this name." << std::endl
                 << "Skipping setting properties for this data: " << dataNodeName.toStdString() << ".";
      continue;
    }

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

      QVariant value = this->ParsePropertyValue(propertyValue);
      this->SetNodeProperty(dataNode, propertyName, value);
    }
  }
}


//-----------------------------------------------------------------------------
QVariant PluginActivator::ParsePropertyValue(const QString& propertyValue)
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
    double doubleValue = propertyValue.toDouble(&ok);
    if (ok)
    {
      if (propertyValue.contains('.') || propertyValue.contains('e') || propertyValue.contains('E'))
      {
        propertyTypedValue.setValue(doubleValue);
      }
      else
      {
        int intValue = propertyValue.toInt(&ok);
        if (ok)
        {
          propertyTypedValue.setValue(intValue);
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
            QLevelWindow range;
            range.SetWindowBounds(minValue, maxValue);
            propertyTypedValue.setValue(range);
          }
        }
      }
    }

    if (!ok)
    {
      propertyTypedValue.setValue(propertyValue);
    }
  }

  return propertyTypedValue;
}


//-----------------------------------------------------------------------------
void PluginActivator::SetNodeProperty(mitk::DataNode* node, const QString& propertyName, const QVariant& propertyValue, const QString& rendererName)
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

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_commonapps, niftk::PluginActivator)
#endif
