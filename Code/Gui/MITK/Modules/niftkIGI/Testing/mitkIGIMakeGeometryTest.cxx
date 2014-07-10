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
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 0");
  surface = MakeAWall(1);
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 1");
  surface = MakeAWall(2);
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 2");
  surface = MakeAWall(3);
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 3");
  surface = MakeAWall(4);
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 4");
  surface = MakeAWall(5);
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 6 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 24 ) ,
      ".. Testing MakeAWall 5");

  surface = MakeLaparoscope (baseDirectory + "lap_06_09.rig",
      baseDirectory + "calib.left.handeye.txt",
      baseDirectory + "calib.left.handeye.txt",
      baseDirectory + "calib.left.handeye.txt", false);
  surface->GetVtkPolyData()->Update();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1708 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 960 ) ,
      ".. Testing make laparoscope");

  surface = MakePointer ( baseDirectory + "pointer_cal.rig", "" );
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 677 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 360 ) ,
      ".. Testing make pointer");
  
  surface = MakeReference ( baseDirectory + "reference.rig", "" );
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 677 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 360 ) ,
      ".. Testing make reference");

  surface = MakeReferencePolaris ( baseDirectory + "reference.rig", "" );
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 733 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 472 ) ,
      ".. Testing make referencePolaris");


  surface = MakeXAxes();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make XAxis");
  surface = MakeYAxes();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make YAxis");
  surface = MakeZAxes();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 1 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 2 ) ,
      ".. Testing make ZAxis");


  surface = MakeLapLensAxes();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 5 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 10 ) ,
      ".. Testing make laplensaxes");

  surface = MakeOptotrak();
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 278 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 380 ) ,
      ".. Testing make Optotrak");

  surface = MakeTransrectalUSProbe("");
  MITK_TEST_CONDITION_REQUIRED(
      ( surface->GetVtkPolyData()->GetNumberOfCells() == 3127 ) &&
      ( surface->GetVtkPolyData()->GetNumberOfPoints() == 1887 ) ,
      ".. Testing make trans rectal US Probe");
  return EXIT_SUCCESS;
}
