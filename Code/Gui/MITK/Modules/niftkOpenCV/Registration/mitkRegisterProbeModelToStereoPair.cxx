/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkRegisterProbeModelToStereoPair.h"
#include <mitkCameraCalibrationFacade.h>
#include "mitkRegistrationHelper.h"
#include "mitkStereoImageToModelSSD.h"
#include <cv.h>
#include <highgui.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <itkAmoebaOptimizer.h>
#include <itkPowellOptimizer.h>

namespace mitk {

//-----------------------------------------------------------------------------
RegisterProbeModelToStereoPair::RegisterProbeModelToStereoPair()
{

}


//-----------------------------------------------------------------------------
RegisterProbeModelToStereoPair::~RegisterProbeModelToStereoPair()
{

}


//-----------------------------------------------------------------------------
bool RegisterProbeModelToStereoPair::DoRegistration(
    const std::string& input3DModelFileName,
    const std::string& inputLeftImageFileName,
    const std::string& inputRightImageFileName,
    const std::string& outputLeftImageFileName,
    const std::string& outputRightImageFileName,
    const std::string& intrinsicLeftFileName,
    const std::string& distortionLeftFileName,
    const std::string& rotationLeftFileName,
    const std::string& translationLeftFileName,
    const std::string& intrinsicRightFileName,
    const std::string& distortionRightFileName,
    const std::string& rightToLeftRotationFileName,
    const std::string& rightToLeftTranslationFileName,
    const float& rx,
    const float& ry,
    const float& rz,
    const float& tx,
    const float& ty,
    const float& tz
    )
{
  bool isSuccessful = false;

  try
  {
    StereoImageToModelSSD::Pointer ssd = StereoImageToModelSSD::New();
    ssd->SetInput3DModelFileName(input3DModelFileName);
    ssd->SetInputLeftImageFileName(inputLeftImageFileName);
    ssd->SetInputRightImageFileName(inputRightImageFileName);
    ssd->SetOutputLeftImageFileName(outputLeftImageFileName);
    ssd->SetOutputRightImageFileName(outputRightImageFileName);
    ssd->SetIntrinsicLeftFileName(intrinsicLeftFileName);
    ssd->SetDistortionLeftFileName(distortionLeftFileName);
    ssd->SetRotationLeftFileName(rotationLeftFileName);
    ssd->SetTranslationLeftFileName(translationLeftFileName);
    ssd->SetIntrinsicRightFileName(intrinsicRightFileName);
    ssd->SetDistortionRightFileName(distortionRightFileName);
    ssd->SetRightToLeftRotationFileName(rightToLeftRotationFileName);
    ssd->SetRightToLeftTranslationFileName(rightToLeftTranslationFileName);
    ssd->SetDrawOutput(true);
    ssd->SetDebug(true);

    ssd->Initialize();

    StereoImageToModelSSD::ParametersType params;
    params.SetSize(6);

    params[0] = rx;
    params[1] = ry;
    params[2] = rz;
    params[3] = tx;
    params[4] = ty;
    params[5] = tz;

    double value = ssd->GetValue(params);
    std::cerr << "Matt, initial value=" << value << std::endl;

    StereoImageToModelSSD::ParametersType scales = params;
    scales.Fill(1);

    StereoImageToModelSSD::ParametersType delta = params;
    delta.Fill(20);

    ssd->SetDrawOutput(false);

    /*
    itk::AmoebaOptimizer::Pointer optimizer = itk::AmoebaOptimizer::New();
    optimizer->SetCostFunction(ssd);
    optimizer->MaximizeOff();
    optimizer->SetScales(scales);
    optimizer->SetInitialSimplexDelta(delta);
    optimizer->SetInitialPosition(params);
    optimizer->SetMaximumNumberOfIterations(10000);
    optimizer->SetParametersConvergenceTolerance(0.01);
    optimizer->StartOptimization();
    */

    itk::PowellOptimizer::Pointer optimizer = itk::PowellOptimizer::New();
    optimizer->MaximizeOff();
    optimizer->SetCostFunction(ssd);
    optimizer->SetMaximumIteration(10000);
    optimizer->SetMaximumLineIteration(1000);
    optimizer->SetInitialPosition(params);
    optimizer->SetScales(scales);
    optimizer->SetValueTolerance(0.0001);
    optimizer->StartOptimization();


    params = ssd->GetParameters();

    std::cerr << "Matt, output position value=" << params[0] \
        << ", " << params[1] \
        << ", " << params[2] \
        << ", " << params[3] \
        << ", " << params[4] \
        << ", " << params[5] \
        << std::endl;

    ssd->SetDrawOutput(false);
    value = ssd->GetValue(params);
    std::cerr << "Matt, final value=" << value << std::endl;

    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "RegisterProbeModelToStereoPair::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
