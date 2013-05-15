/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <niftkMakeLaparoscopicUSProbeCLP.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkAppendPolyData.h>
#include <vtkCylinderSource.h>
#include <vtkPolyDataWriter.h>

/**
 * \brief Generates a VTK model to represent an ultrasound probe.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( outputVTKSurface.length() == 0
      )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
  sphereSource->SetRadius(radius);
  sphereSource->SetCenter(0, radius, 0);
  sphereSource->SetThetaResolution(resolution);
  sphereSource->SetPhiResolution(resolution);
  sphereSource->SetStartTheta(180);
  sphereSource->SetEndTheta(360);

  vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkSmartPointer<vtkCylinderSource>::New();
  cylinderSource->SetCenter(0, ((length-radius)/2.0) + radius, 0);
  cylinderSource->SetRadius(radius);
  cylinderSource->SetHeight(length-radius);
  cylinderSource->SetResolution(resolution);
  cylinderSource->SetCapping(false);

  vtkSmartPointer<vtkAppendPolyData> appender = vtkSmartPointer<vtkAppendPolyData>::New();
  appender->AddInput(sphereSource->GetOutput());
  appender->AddInput(cylinderSource->GetOutput());

  vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
  writer->SetInput(appender->GetOutput());
  writer->SetFileName(outputVTKSurface.c_str());
  writer->Update();

  return EXIT_SUCCESS;
}
