/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "niftkMakeLaparoscopicUSProbeCLP.h"
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
