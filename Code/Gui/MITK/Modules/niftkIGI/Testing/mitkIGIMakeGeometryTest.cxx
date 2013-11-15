/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkMakeGeometry.h>

#include <niftkVTKFunctions.h>
#include <vtkSmartPointer.h>

namespace mitk
{

} // end namespace

int mitkIGIMakeGeometryTest(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: mitkIGIMakeGeometryTest directory" << std::endl;
    std::cerr << " argc=" << argc << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << " argv[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  } 
  
  std::string baseDirectory = argv[1];
  mitk::Surface::Pointer surface = mitk::Surface::New();

  surface = MakeAWall(0);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 0");
  surface = MakeAWall(1);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 1");
  surface = MakeAWall(2);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 2");
  surface = MakeAWall(3);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 3");
  surface = MakeAWall(4);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 4");
  surface = MakeAWall(5);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 5");

  surface = MakeLaparoscope (baseDirectory + "lap_06_09.rig", baseDirectory + "calib.left.handeye.txt");
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1387 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 862 ) ,
      ".. Testing make laparoscope");

  surface = MakePointer ( baseDirectory + "pointer_cal.rig", "" );
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 677 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 360 ) ,
      ".. Testing make pointer");
  surface = MakeReference ( baseDirectory + "reference.rig", "" );
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 677 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 360 ) ,
      ".. Testing make reference");

  surface = MakeXAxes();
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make XAxis");
  surface = MakeYAxes();
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make YAxis");
  surface = MakeZAxes();
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make ZAxis");


  surface = MakeLapLensAxes();
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 5 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 10 ) ,
      ".. Testing make laplensaxes");

  surface = MakeOptotrak();
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 278 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 380 ) ,
      ".. Testing make Optotrak");

  surface = MakeTransrectalUSProbe("");
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 3127 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 1887 ) ,
      ".. Testing make trans rectal US Probe");
  return EXIT_SUCCESS;
}
