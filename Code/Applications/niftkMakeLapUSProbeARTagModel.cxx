/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkMakeLapUSProbeARTagModelCLP.h>
#include <cmath>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkFunctions.h>
#include <vtkSphereSource.h>
#include <vtkAppendPolyData.h>
#include <vtkCylinderSource.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>

/**
 * \brief Generates a VTK model to match the ARTag board created by aruco_create_board.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( outputTrackingModel.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  double pi = 3.14159265358979;
  double mmPerInch = 25.4;
  int printerDotsPerInch = 300;

  double printerDotsPerMillimetre = (printerDotsPerInch / mmPerInch);
  int numberOfPixelsAlongLength = (int)(length*printerDotsPerMillimetre);

  // numberSquares * pixelsPerTagSquare + (numberSquares - 1)*pixelsPerTagSquare*0.2 = numberOfPixelsAlongLength
  // numberSquares * pixelsPerTagSquare + numberSquares*pixelsPerTagSquare*0.2 - pixelsPerTagSquare*0.2  = numberOfPixelsAlongLength
  // pixelsPerTagSquare*(numberSquares + numberSquares*0.2 - 0.2) = numberOfPixelsAlongLength
  // pixelsPerTagSquare = numberOfPixelsAlongLength / (numberSquares + numberSquares*0.2 - 0.2)

  int pixelsPerTagSquare = static_cast<int>(numberOfPixelsAlongLength / (numberSquares + numberSquares*0.2 - 0.2));
  int actualLengthInPixels = static_cast<int>(pixelsPerTagSquare*numberSquares + (numberSquares-1)*(pixelsPerTagSquare*0.2));
  double actualLengthInMillimetres = actualLengthInPixels / printerDotsPerMillimetre;

  double maxCircumferenceInMillimetres = pi * diameter * ((pi - asin(surface/diameter))/pi); // distance around probe; ie. section of circumference
  int maxCircumferenceInPixels = static_cast<int>(maxCircumferenceInMillimetres * printerDotsPerMillimetre);
  int numberSquaresAlongWidth = static_cast<int>(maxCircumferenceInPixels / (pixelsPerTagSquare*1.2));
  int actualWidthInPixels = static_cast<int>(numberSquaresAlongWidth*pixelsPerTagSquare + (numberSquaresAlongWidth-1)*(pixelsPerTagSquare*0.2));
  double actualWidthInMillimetres = actualWidthInPixels / printerDotsPerMillimetre;
  double tagSquareSizeInMillimetres = pixelsPerTagSquare / printerDotsPerMillimetre;

  std::cout << "dots per inch to print at   = " << printerDotsPerInch << std::endl;
  std::cout << "dots per millimetres        = " << printerDotsPerMillimetre << std::endl;
  std::cout << "length in millimetres       = " << length << std::endl;
  std::cout << "required number squares     = " << numberSquares << std::endl;
  std::cout << "number of pixels per square = " << pixelsPerTagSquare << std::endl;
  std::cout << "size of square in (mm)      = " << tagSquareSizeInMillimetres << std::endl;
  std::cout << "max circ. millimetres       = " << maxCircumferenceInMillimetres << std::endl;
  std::cout << "max circ. pixels            = " << maxCircumferenceInPixels << std::endl;
  std::cout << "number squares along width  = " << numberSquaresAlongWidth << std::endl;
  std::cout << "actual width in pixels      = " << actualWidthInPixels << std::endl;
  std::cout << "actual length in pixels     = " << actualLengthInPixels << std::endl;
  std::cout << "actual width (mm)           = " << actualWidthInMillimetres << std::endl;
  std::cout << "actual length (mm)          = " << actualLengthInMillimetres << std::endl << std::endl;

  if (inputPointIDs.length() == 0)
  {
    std::cout << "Run: aruco_create_board " << numberSquaresAlongWidth << ":" << numberSquares << " board.png board.yml " << pixelsPerTagSquare << std::endl;
    std::cout << "Print at: " <<  printerDotsPerInch << " dpi" << std::endl << std::endl;
    std::cout << "Then extract pointIDs from .yml file with: cat board.yml | grep id | cut -f 2 -d \":\" | cut -f 1 -d \",\" > pointIDs.txt " << std::endl;
    std::cout << "Then re-run this program passing in pointIDs.txt with: --inputPointIDs pointIDs.txt" << std::endl;
    return EXIT_SUCCESS;
  }

  double circumferenceInMillimetres = pi*diameter;
  double proportionOfCircle = actualWidthInMillimetres/circumferenceInMillimetres;
  double angleAroundCircle = 2*pi*proportionOfCircle;
  double angleNotAroundCircle = 2*pi - angleAroundCircle;
  double halfAngleNotAroundCircle = angleNotAroundCircle/2.0;
  double radius = diameter/2.0;
  double offsetOfOriginFromCentre = 1.0 * radius * cos(halfAngleNotAroundCircle);
  double offSetOfOrigin = radius - offsetOfOriginFromCentre;

  std::cout << "radius                      = " << radius << std::endl;
  std::cout << "offsetOfOriginFromCentre    = " << offsetOfOriginFromCentre << std::endl;
  std::cout << "offSetOfOrigin              = " << offSetOfOrigin << std::endl << std::endl;

  std::vector<int> pointIDs;
  if(inputPointIDs.size() > 0)
  {
    ifstream myfile(inputPointIDs.c_str());
    if (myfile.is_open())
    {
      while(!myfile.eof() && (int)pointIDs.size() < numberSquaresAlongWidth*numberSquares)
      {
        int tmp;
        myfile >> tmp;
        pointIDs.push_back(tmp);
      }
    }
  }
  if ((int)pointIDs.size() != numberSquaresAlongWidth*numberSquares)
  {
    std::cerr << "ERROR: Incorrect number of Point IDs supplied. Got " << pointIDs.size() << " but was expecting " << numberSquaresAlongWidth*numberSquares << std::endl;
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::New();

  vtkPoints *points = vtkPoints::New();
  points->SetDataTypeToDouble();
  points->Initialize();

  vtkSmartPointer<vtkDoubleArray> normals = vtkDoubleArray::New();
  normals->SetNumberOfComponents(3);
  normals->SetName("Normals");
  normals->Initialize();

  vtkSmartPointer<vtkIntArray> pointIDArray = vtkIntArray::New();
  pointIDArray->SetNumberOfComponents(1);
  pointIDArray->SetName("Point IDs");
  pointIDArray->Initialize();

  vtkSmartPointer<vtkCellArray> vertices = vtkCellArray::New();
  vertices->Initialize();

  int pointCounter = 0;
  for (int j = 0; j < numberSquares; ++j)
  {
    for (int i = numberSquaresAlongWidth-1; i >= 0; --i)
    {
      double distanceAroundCircumference = (0.5 + i*1.2)*tagSquareSizeInMillimetres;
      double proportionAroundCircumference = distanceAroundCircumference/actualWidthInMillimetres;

      double normal[3];
      double normalised[3];
      double centre[3];
      double point[3];

      point[0] = radius * sin(halfAngleNotAroundCircle + proportionAroundCircumference*angleAroundCircle);
      point[1] = radius * cos(pi + halfAngleNotAroundCircle + proportionAroundCircumference*angleAroundCircle) + radius - offSetOfOrigin;
      point[2] = (0.5 + j*1.2)*tagSquareSizeInMillimetres;

      centre[0] = 0;
      centre[1] = offSetOfOrigin;
      centre[2] = point[2];

      normal[0] = point[0] - centre[0];
      normal[1] = point[1] - centre[1];
      normal[2] = point[2] - centre[2];

      NormaliseToUnitLength(normal, normalised);

      points->InsertNextPoint(point[0], point[1], point[2]);
      normals->InsertNextTuple3(normalised[0], normalised[1], normalised[2]);
      pointIDArray->InsertNextTuple1(pointIDs[pointCounter]);
      pointCounter++;

      vertices->InsertNextCell(1);
      vertices->InsertCellPoint(points->GetNumberOfPoints() - 1);
    }
  }

  polyData->SetPoints(points);
  polyData->SetVerts(vertices);
  polyData->GetPointData()->SetNormals(normals);
  polyData->GetPointData()->SetScalars(pointIDArray);

  vtkSmartPointer<vtkPolyDataWriter> polyWriter = vtkPolyDataWriter::New();
  polyWriter->SetFileName(outputTrackingModel.c_str());
  polyWriter->SetInput(polyData);
  polyWriter->SetFileTypeToASCII();
  polyWriter->Write();

  std::cout << "written tracking model to      = " << outputTrackingModel << std::endl;

  // So, if outputVisualisationModel is supplied we also output a VTK surface model just
  // for overlay purposes onto a video overlay. Here we just create a cyclinder.
  if (outputVisualisationModel.size() > 0)
  {
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSphereSource::New();
    sphereSource->SetRadius(radius);
    sphereSource->SetCenter(0, offsetOfOriginFromCentre, tipZOffset);
    sphereSource->SetThetaResolution(36);
    sphereSource->SetPhiResolution(36);
    sphereSource->SetStartPhi(90);
    sphereSource->SetEndPhi(180);

    vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkCylinderSource::New();
    cylinderSource->SetCenter(0, length/2.0, -offsetOfOriginFromCentre);
    cylinderSource->SetRadius(radius);
    cylinderSource->SetHeight(length);
    cylinderSource->SetResolution(36);
    cylinderSource->SetCapping(false);

    vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
    transform->Identity();
    transform->RotateX(90);

    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkTransformPolyDataFilter::New();
    transformFilter->SetInput(cylinderSource->GetOutput());
    transformFilter->SetTransform(transform);

    vtkSmartPointer<vtkAppendPolyData> appender = vtkAppendPolyData::New();
    appender->AddInput(sphereSource->GetOutput());
    appender->AddInput(transformFilter->GetOutput());

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();
    writer->SetInput(appender->GetOutput());
    writer->SetFileName(outputVisualisationModel.c_str());
    writer->Update();

    std::cout << "written visualisation model to = " << outputVisualisationModel << std::endl;

    std::cout << "tip origin                     = (0, 0, " << tipZOffset - radius << ")" << std::endl;
  }

  return EXIT_SUCCESS;
}
