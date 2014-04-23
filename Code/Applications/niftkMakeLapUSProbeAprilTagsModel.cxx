/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkMakeLapUSProbeAprilTagsModelCLP.h>
#include <cmath>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <niftkVTKFunctions.h>
#include <vtkSphereSource.h>
#include <vtkCylinderSource.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>
#include <vtkFloatArray.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkDecimatePolylineFilter.h>

void ConvertGridPointToCyclinderPoint(int pointId,
                                      int lengthCounter, int widthCounter,
                                      double originX,
                                      double originY,
                                      double borderSizeInMillimetres,
                                      double tagSizeInMillimetres,
                                      double offsetXInMillimetres,
                                      double offsetYInMillimetres,
                                      double radius,
                                      vtkPoints *points,
                                      vtkDoubleArray *normals,
                                      vtkIntArray *pointIds,
                                      vtkCellArray *vertices
                                      )
{
  double pi = 3.14159265358979;

  double xOnGrid, yOnGrid, proportionOfCircumference, angleAroundCircumferenceInRadians;
  double xOnPolarDiagram, yOnPolarDiagram;
  double cylinderCentre[3];
  double cylinderPoint[3];
  double cylinderNormal[3];
  double circumferenceInMillimetres = pi * radius * 2.0;

  xOnGrid = originX + widthCounter*tagSizeInMillimetres + borderSizeInMillimetres + offsetXInMillimetres;
  yOnGrid = originY + lengthCounter*tagSizeInMillimetres + borderSizeInMillimetres + offsetYInMillimetres;

  proportionOfCircumference = (xOnGrid - originX)/ circumferenceInMillimetres;
  angleAroundCircumferenceInRadians = 2*pi*proportionOfCircumference;

  xOnPolarDiagram = cos(angleAroundCircumferenceInRadians);
  yOnPolarDiagram = sin(angleAroundCircumferenceInRadians);

  cylinderPoint[0] = -radius*(xOnPolarDiagram);
  cylinderPoint[1] = radius*(yOnPolarDiagram);
  cylinderPoint[2] = yOnGrid;
  cylinderCentre[0] = 0;
  cylinderCentre[1] = 0;
  cylinderCentre[2] = yOnGrid;

  niftk::CalculateUnitVector(cylinderPoint, cylinderCentre, cylinderNormal);

  points->InsertNextPoint(cylinderPoint[0], cylinderPoint[1], cylinderPoint[2]);
  normals->InsertNextTuple3(cylinderNormal[0], cylinderNormal[1], cylinderNormal[2]);
  pointIds->InsertNextTuple1(pointId);
  vertices->InsertNextCell(1);
  vertices->InsertCellPoint(points->GetNumberOfPoints() - 1);
}


/**
 * \brief Generates a VTK model to match the April Tag board created by niftkMakeGridOf2DImages.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if (    outputTrackingModel.length() == 0
       || outputVisualisationModel.length() == 0
       || textureMap.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  double pi = 3.14159265358979;
  double mmPerInch = 25.4;
  int printerDotsPerInch = 300;

  double radius = diameter / 2.0;
  double circumference = pi*diameter;
  double actualTagSizeIncludingBorder = outputTagSize*9/static_cast<double>(7);
  double borderSizeInMillimetres = actualTagSizeIncludingBorder / static_cast<double>(9);
  double cornerOffsetInMillimetres = borderSizeInMillimetres * 7;
  double centreOffsetInMillimetres = borderSizeInMillimetres * 3.5;
  double spacingBetweenTags = borderSizeInMillimetres * 2;

  double printerDotsPerMillimetre = (printerDotsPerInch / mmPerInch);
  int numberOfAvailablePixelsAlongLength = (int)(length*printerDotsPerMillimetre);
  int numberOfTagsAlongLength = std::floor(static_cast<double>(length)/static_cast<double>(actualTagSizeIncludingBorder));
  int numberOfPixelsRequiredAlongLength = numberOfTagsAlongLength*inputTagSize;

  double maxCircumferenceInMillimetres = pi * diameter * ((pi - asin(surface/diameter))/pi); // distance around probe; ie. section of circumference
  int numberTagsAlongWidth = std::floor(static_cast<double>(maxCircumferenceInMillimetres)/static_cast<double>(actualTagSizeIncludingBorder));
  int numberOfPixelsRequiredAlongWidth = numberTagsAlongWidth*inputTagSize;

  double minXInMillimetres = 0 - ((numberTagsAlongWidth*actualTagSizeIncludingBorder)/2.0);
  double minYInMillimetres = 0;
  double maxXInMillimetres = minXInMillimetres + (numberTagsAlongWidth*actualTagSizeIncludingBorder);
  double maxYInMillimetres = minYInMillimetres + (numberOfTagsAlongLength*actualTagSizeIncludingBorder);

  double cyclinderLength = numberOfTagsAlongLength*actualTagSizeIncludingBorder;

  std::cout << "dots per inch to print at      = " << printerDotsPerInch << std::endl;
  std::cout << "dots per millimetres           = " << printerDotsPerMillimetre << std::endl;
  std::cout << "number of pixels available     = " << numberOfAvailablePixelsAlongLength << std::endl;
  std::cout << "number of tags along length    = " << numberOfTagsAlongLength << std::endl;
  std::cout << "circumference in millimetres   = " << maxCircumferenceInMillimetres << std::endl;
  std::cout << "number of tags along width     = " << numberTagsAlongWidth << std::endl;
  std::cout << "border size in millimetres     = " << borderSizeInMillimetres << std::endl;
  std::cout << "corner offset in millimetres   = " << cornerOffsetInMillimetres << std::endl;
  std::cout << "centre offset in millimetres   = " << centreOffsetInMillimetres << std::endl;
  std::cout << "spacing between tags (mm)      = " << spacingBetweenTags << std::endl;
  std::cout << "Tag Size To Print (pixels)     = " << numberOfPixelsRequiredAlongWidth << " x " << numberOfPixelsRequiredAlongLength << std::endl;
  std::cout << "Tag Size To Print (mm)         = " << numberTagsAlongWidth*actualTagSizeIncludingBorder << " x " << numberOfTagsAlongLength*actualTagSizeIncludingBorder << std::endl;
  std::cout << "Board region in millimetres    = " << minXInMillimetres << ", " << minYInMillimetres << " to " << maxXInMillimetres << ", " << maxYInMillimetres << std::endl;
  std::cout << "Cylinder length                = " << cyclinderLength << std::endl;
  std::cout << "Cylinder circumference         = " << circumference << std::endl;
  std::cout << "Cylinder radius                = " << radius << std::endl;

  // Aim is:
  // 1. to make wrap the coordinates of the corner of each tag, and the centre of each tag around a cylinder.
  // 2. We also want surface normals.
  // 3. Z axis = along the probe.
  // 4. Right hand coordinate system.
  // 5. Origin is the centre of the probe, aligned with the face .... i.e. NOT including the white border.

  vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::New();

  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
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

  for (int lengthCounter = 0; lengthCounter < numberOfTagsAlongLength; lengthCounter++)
  {
    for (int widthCounter = 0; widthCounter < numberTagsAlongWidth; widthCounter++)
    {
      int pointID = widthCounter + lengthCounter*numberTagsAlongWidth;

      ConvertGridPointToCyclinderPoint(pointID+0,     lengthCounter, widthCounter, minXInMillimetres, minYInMillimetres, borderSizeInMillimetres, actualTagSizeIncludingBorder, centreOffsetInMillimetres, centreOffsetInMillimetres, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+10000, lengthCounter, widthCounter, minXInMillimetres, minYInMillimetres, borderSizeInMillimetres, actualTagSizeIncludingBorder, 0,                         0,                         radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+20000, lengthCounter, widthCounter, minXInMillimetres, minYInMillimetres, borderSizeInMillimetres, actualTagSizeIncludingBorder, cornerOffsetInMillimetres, 0,                         radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+30000, lengthCounter, widthCounter, minXInMillimetres, minYInMillimetres, borderSizeInMillimetres, actualTagSizeIncludingBorder, cornerOffsetInMillimetres, cornerOffsetInMillimetres, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+40000, lengthCounter, widthCounter, minXInMillimetres, minYInMillimetres, borderSizeInMillimetres, actualTagSizeIncludingBorder, 0,                         cornerOffsetInMillimetres, radius, points, normals, pointIDArray, vertices);
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
  // for overlay purposes onto a video overlay. Here we just create an open ended cyclinder.

  if (outputVisualisationModel.size() > 0)
  {
    // Generate an open ended cylinder, along +z axis.
    // Note each quad is completely independant.

    vtkSmartPointer<vtkPolyData> polyData2 = vtkPolyData::New();

    vtkSmartPointer<vtkPoints> points2 = vtkPoints::New();
    points2->SetDataTypeToDouble();
    points2->Initialize();

    vtkSmartPointer<vtkIntArray> pointIDArray2 = vtkIntArray::New();
    pointIDArray2->SetNumberOfComponents(1);
    pointIDArray2->SetName("Point IDs");
    pointIDArray2->Initialize();

    vtkSmartPointer<vtkCellArray> quads = vtkCellArray::New();
    quads->Initialize();

    vtkIdType pointsFor1Quad[4];
    double cylinderPoint[3];

    for (vtkIdType lengthCounter = 0; lengthCounter < pointsAlongLength-1; lengthCounter++)
    {
      for (vtkIdType circumferenceCounter = 0; circumferenceCounter < pointsAroundCircumference-1; circumferenceCounter++)
      {
        double thetaAtStartOfQuad = 2*pi*circumferenceCounter/static_cast<double>(pointsAroundCircumference-1);
        double xOnPolarDiagramStart = cos(thetaAtStartOfQuad);
        double yOnPolarDiagramStart = sin(thetaAtStartOfQuad);

        double thetaAtEndOfQuad = 2*pi*(circumferenceCounter+1)/static_cast<double>(pointsAroundCircumference-1);
        double xOnPolarDiagramEnd = cos(thetaAtEndOfQuad);
        double yOnPolarDiagramEnd = sin(thetaAtEndOfQuad);

        double zStart = cyclinderLength*lengthCounter/static_cast<double>(pointsAlongLength-1);
        double zEnd = cyclinderLength*(lengthCounter+1)/static_cast<double>(pointsAlongLength-1);

        cylinderPoint[0] = -radius*(xOnPolarDiagramStart);
        cylinderPoint[1] = radius*(yOnPolarDiagramStart);
        cylinderPoint[2] = zStart;
        pointsFor1Quad[0] = points2->InsertNextPoint(cylinderPoint[0], cylinderPoint[1], cylinderPoint[2]);
        pointIDArray2->InsertNextTuple1(pointsFor1Quad[0]);

        cylinderPoint[0] = -radius*(xOnPolarDiagramEnd);
        cylinderPoint[1] = radius*(yOnPolarDiagramEnd);
        cylinderPoint[2] = zStart;
        pointsFor1Quad[1] = points2->InsertNextPoint(cylinderPoint[0], cylinderPoint[1], cylinderPoint[2]);
        pointIDArray2->InsertNextTuple1(pointsFor1Quad[1]);

        cylinderPoint[0] = -radius*(xOnPolarDiagramEnd);
        cylinderPoint[1] = radius*(yOnPolarDiagramEnd);
        cylinderPoint[2] = zEnd;
        pointsFor1Quad[2] = points2->InsertNextPoint(cylinderPoint[0], cylinderPoint[1], cylinderPoint[2]);
        pointIDArray2->InsertNextTuple1(pointsFor1Quad[2]);

        cylinderPoint[0] = -radius*(xOnPolarDiagramStart);
        cylinderPoint[1] = radius*(yOnPolarDiagramStart);
        cylinderPoint[2] = zEnd;
        pointsFor1Quad[3] = points2->InsertNextPoint(cylinderPoint[0], cylinderPoint[1], cylinderPoint[2]);
        pointIDArray2->InsertNextTuple1(pointsFor1Quad[3]);

        quads->InsertNextCell(4, pointsFor1Quad);
      }
    }

    // Now generate texture coords for each point on visualisation model.

    vtkIdType numberPoints = points2->GetNumberOfPoints();

    vtkFloatArray* tc = vtkFloatArray::New();
    tc->SetNumberOfComponents( 2 );
    tc->Allocate(numberPoints);

    vtkSmartPointer<vtkDoubleArray> normals2 = vtkDoubleArray::New();
    normals2->SetNumberOfComponents(3);
    normals2->SetName("Normals");
    normals2->Initialize();

    for (vtkIdType counter = 0; counter < numberPoints; counter++)
    {
      double *p = points2->GetPoint(counter);

      // Need to measure distance around curvature of probe.
      double norm[2] = {0, 0};
      norm[0] = p[0];
      norm[1] = p[1];
      double distance = sqrt(norm[0]*norm[0] + norm[1]*norm[1]);
      norm[0] /= distance;
      norm[1] /= distance;
      double cosTheta = -norm[0];
      double theta = acos(cosTheta);
      if (norm[1] < 0)
      {
        theta = pi + (pi-theta);
      }
      double portionOfCircle = theta/(2*pi);

      double dx = portionOfCircle*circumference;

      // z axis in space, maps to y axis in texture map. So distance along z = how far along texture map we travel in y direction.
      double dy = p[2];

      if (dx > (numberTagsAlongWidth*actualTagSizeIncludingBorder))
      {
        dx = 1*borderSizeInMillimetres;
        dy = 1*borderSizeInMillimetres;
      }

      double tx = dx / (maxXInMillimetres - minXInMillimetres);
      double ty = -dy / (maxYInMillimetres - minYInMillimetres);

      // Convert to texture coord.
      tc->InsertNextTuple2(tx, ty);
      normals2->InsertNextTuple3(norm[0], norm[1], 0);
    }

    polyData2->SetPoints(points2);
    polyData2->SetPolys(quads);
    polyData2->GetPointData()->SetNormals(normals2);
    polyData2->GetPointData()->SetTCoords(tc);

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();
    writer->SetInput(polyData2);
    writer->SetFileName(outputVisualisationModel.c_str());
    writer->Update();

    std::cout << "written visualisation model to = " << outputVisualisationModel << std::endl;
  }

  return EXIT_SUCCESS;
}
