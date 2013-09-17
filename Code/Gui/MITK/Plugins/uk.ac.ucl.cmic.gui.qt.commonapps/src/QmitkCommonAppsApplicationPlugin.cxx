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

#include <mitkProperties.h>
#include <mitkVersion.h>
#include <mitkLogMacros.h>
#include <mitkDataStorage.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkImageAccessByItk.h>
#include <itkStatisticsImageFilter.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <QFileInfo>
#include <QDateTime>
#include <QtPlugin>

#include <NifTKConfigure.h>
//#include <mitkMIDASTool.h>
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
  this->RegisterQmitkCommonAppsExtensions();
  this->RegisterDataStorageListener();
  this->BlankDepartmentalLogo();
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
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsApplicationPlugin::UnregisterDataStorageListener()
{
  if (m_DataStorageServiceTracker != NULL)
  {

    this->GetDataStorage()->AddNodeEvent.RemoveListener
        ( mitk::MessageDelegate1<QmitkCommonAppsApplicationPlugin, const mitk::DataNode*>
          ( this, &QmitkCommonAppsApplicationPlugin::NodeAddedProxy ) );

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
void QmitkCommonAppsApplicationPlugin::RegisterQmitkCommonAppsExtensions()
{
  BERRY_REGISTER_EXTENSION_CLASS(QmitkCommonAppsApplicationPreferencePage, m_Context);
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
  if(!m_InDataStorageChanged)
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
  this->RegisterBlackOpacityProperty("uk.ac.ucl.cmic.gui.qt.commonapps", node);
  this->RegisterBinaryImageProperties("uk.ac.ucl.cmic.gui.qt.commonapps", node);
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
void QmitkCommonAppsApplicationPlugin::RegisterMIDASGlobalInteractionPatterns()
{
  mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();
/*
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_DROPPER_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_DRAW_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_POLY_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_KEYPRESS_STATE_MACHINE_XML);
*/
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
void QmitkCommonAppsApplicationPlugin::RegisterBlackOpacityProperty(
    const std::string& preferencesNodeName, mitk::DataNode *node)
{
  if (mitk::IsNodeAGreyScaleImage(node))
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    berry::IPreferences* prefNode = this->GetPreferencesNode(preferencesNodeName);

    if (prefNode != NULL && image.IsNotNull())
    {
      bool blackOpacity = prefNode->GetBool(QmitkCommonAppsApplicationPreferencePage::BLACK_OPACITY, true);

      if (blackOpacity)
      {
        node->SetProperty("black opacity", mitk::FloatProperty::New(1));
      }
      else
      {
        node->SetProperty("black opacity", mitk::FloatProperty::New(0));
      }

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
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_commonapps, QmitkCommonAppsApplicationPlugin)
