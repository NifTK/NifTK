/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMakeLapUSProbeSimulationDataCLP.h"

#include <QApplication>
#include <QSizePolicy>
#include <mitkPoint.h>
#include <mitkVector.h>
#include <QmitkCalibratedModelRenderingPipeline.h>

/**
 * \brief Generates Simulation Data to mimic tag tracking.
 */
int main(int argc, char** argv)
{
  int returnStatus = EXIT_FAILURE;

  try
  {
    // To parse command line args.
    PARSE_ARGS;

    if (   modelForTracking.length() == 0
        || modelForVisualisation.length() == 0
        || texture.length() == 0
        || leftIntrinsics.length() == 0
        || rightIntrinsics.length() == 0
        || rightToLeft.length() == 0
        || outputData.length() == 0
        )
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus;
    }

    mitk::Point2I ws;
    ws[0] = windowSize[0];
    ws[1] = windowSize[1];

    mitk::Point2I cws;
    cws[0] = calibratedWindowSize[0];
    cws[1] = calibratedWindowSize[1];

    QApplication app(argc,argv);

    QmitkCalibratedModelRenderingPipeline pl("niftkMakeLapUSProbeSimulationData", ws, cws, leftIntrinsics, rightIntrinsics, modelForVisualisation, rightToLeft, texture, modelForTracking, ultrasoundCalibration, ultrasoundImage, radius, outputData);
    pl.setGeometry(0, 0, ws[0], ws[1]);
    pl.setMinimumWidth(ws[0]);
    pl.setMaximumWidth(ws[0]);
    pl.setMinimumHeight(ws[1]);
    pl.setMaximumHeight(ws[1]);
    pl.setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    pl.SetIsRightHandCamera(false);
    pl.SetModelToWorldTransform(modelToWorldTransformation);
    pl.SetWorldToCameraTransform(worldToCameraTransformation);

    pl.show();
    pl.Render();
    pl.SaveData();

    returnStatus = EXIT_SUCCESS;
    //returnStatus = app.exec();
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = EXIT_FAILURE + 3;
  }
  return returnStatus;
}
