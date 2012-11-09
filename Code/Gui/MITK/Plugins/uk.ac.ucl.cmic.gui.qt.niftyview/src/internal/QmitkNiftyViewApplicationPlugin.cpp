/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-05 06:46:30 +0000 (Sat, 05 Nov 2011) $
 Revision          : $Revision: 7703 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewApplicationPlugin.h"
#include "QmitkNiftyViewIGIPerspective.h"
#include "QmitkNiftyViewMIDASPerspective.h"
#include "QmitkNiftyViewApplicationPreferencePage.h"
#include "../QmitkNiftyViewApplication.h"

#include <mitkVersion.h>
#include <mitkLogMacros.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkLevelWindow.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkGlobalInteraction.h>

#include <service/cm/ctkConfigurationAdmin.h>
#include <service/cm/ctkConfiguration.h>

#include <QFileInfo>
#include <QDateTime>
#include <QtPlugin>

#include "NifTKConfigure.h"
#include "mitkMIDASTool.h"

#include "itkStatisticsImageFilter.h"

//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPlugin::QmitkNiftyViewApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyViewApplicationPlugin::~QmitkNiftyViewApplicationPlugin()
{
}


//-----------------------------------------------------------------------------
QString QmitkNiftyViewApplicationPlugin::GetHelpHomePageURL() const
{
  return QString("qthelp://uk.ac.ucl.cmic.gui.qt.niftyview/bundle/index.html");
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewApplicationPlugin::start(ctkPluginContext* context)
{
  QmitkCommonAppsApplicationPlugin::start(context);
  
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplication, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewIGIPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewMIDASPerspective, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkNiftyViewApplicationPreferencePage, context);

  // Load StateMachine patterns
  mitk::GlobalInteraction* globalInteractor =  mitk::GlobalInteraction::GetInstance();
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_DROPPER_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_DRAW_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_POLY_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML);
  globalInteractor->GetStateMachineFactory()->LoadBehaviorString(mitk::MIDASTool::MIDAS_KEYPRESS_STATE_MACHINE_XML);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
QmitkNiftyViewApplicationPlugin
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
void QmitkNiftyViewApplicationPlugin::NodeAdded(const mitk::DataNode *constNode)
{
  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(constNode->GetData());
  if (image.IsNotNull())
  {
    bool isBinary(true);
    constNode->GetBoolProperty("binary", isBinary);

    if (isBinary)
    {
      return;
    }

    berry::IPreferencesService::Pointer prefService =
    berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

    berry::IPreferences::Pointer prefNode = prefService->GetSystemPreferences()->Node("uk.ac.ucl.cmic.gui.qt.niftyview");
    double percentageOfRange = prefNode->GetDouble(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_PERCENTAGE_NAME, 50);
    std::string initialisationMethod = prefNode->Get(QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_METHOD_NAME, QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS);

    mitk::DataNode::Pointer node = const_cast<mitk::DataNode*>(constNode);

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
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception during QmitkNiftyViewApplicationPlugin::ITKGetStatistics, so image statistics will be wrong." << e.what();
      }
    }

    if (!minDataLimitFound || !maxDataLimitFound || !meanDataFound || !stdDevDataFound)
    {
      double windowMin = 0;
      double windowMax = 0;
      mitk::LevelWindow levelWindow;

      // This image hasn't had the data members that this view needs (minDataLimit, maxDataLimit etc) initialized yet.
      // i.e. we haven't seen it before. So we have a choice of how to initialise the Level/Window.
      if (initialisationMethod == QmitkNiftyViewApplicationPreferencePage::IMAGE_INITIALISATION_MIDAS)
      {
        double centre = (minDataLimit + 4.51*stdDevData)/2.0;
        double width = 4.5*stdDevData;
        windowMin = centre - width/2.0;
        windowMax = centre + width/2.0;
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

      levelWindow.SetRangeMinMax(minDataLimit, maxDataLimit);
      levelWindow.SetWindowBounds(windowMin, windowMax);
      node->SetLevelWindow(levelWindow);
    }
  }
}

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_gui_qt_niftyview, QmitkNiftyViewApplicationPlugin)
