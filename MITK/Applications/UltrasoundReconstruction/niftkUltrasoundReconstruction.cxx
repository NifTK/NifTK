/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkUltrasoundReconstructionCLP.h>
#include <niftkUltrasoundProcessing.h>
#include <mitkVector.h>
#include <mitkExceptionMacro.h>
#include <mitkIOUtil.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkMITKMathsUtils.h>
#include <niftkFileIOUtils.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    imageDirectory.length() == 0
       || matrixDirectory.length() == 0
       || rigidMatrixFile.length() == 0
       || scalingMatrixFile.length() == 0
       || outputImage.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    niftk::TrackedImageData data = niftk::LoadImageAndTrackingDataFromDirectories(imageDirectory, matrixDirectory);
    vtkSmartPointer<vtkMatrix4x4> rigidMatrix = niftk::LoadVtkMatrix4x4FromFile(rigidMatrixFile);
    vtkSmartPointer<vtkMatrix4x4> scalingMatrix = niftk::LoadVtkMatrix4x4FromFile(scalingMatrixFile);

    mitk::Point2D scaleFactors;
    scaleFactors[0] = scalingMatrix->GetElement(0, 0);
    scaleFactors[1] = scalingMatrix->GetElement(1, 1);

    niftk::RotationTranslation imageToSensor;
    niftk::ConvertMatrixToRotationAndTranslation(*rigidMatrix, imageToSensor.first, imageToSensor.second);

    mitk::Vector3D spacing;
    spacing[0] = voxelSize[0];
    spacing[1] = voxelSize[1];
    spacing[2] = voxelSize[2];

    mitk::Image::Pointer volume = niftk::DoUltrasoundReconstruction(data,          // input data
                                                                    scaleFactors,  // from calibration
                                                                    imageToSensor, // from calibration
                                                                    spacing);      // command line arg

    mitk::IOUtil::Save(volume, outputImage);
    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}
