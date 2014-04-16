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
                                      double offsetXInMillimetres, double offsetYInMillimetres,
                                      double tagSizeInMillimetres,
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

  xOnGrid = offsetXInMillimetres + widthCounter*tagSizeInMillimetres;
  yOnGrid = offsetYInMillimetres + lengthCounter*tagSizeInMillimetres;
  proportionOfCircumference = xOnGrid / circumferenceInMillimetres;
  angleAroundCircumferenceInRadians = 2*pi*proportionOfCircumference;
  xOnPolarDiagram = cos(angleAroundCircumferenceInRadians);
  yOnPolarDiagram = sin(angleAroundCircumferenceInRadians);
  cylinderPoint[0] = -radius*(xOnPolarDiagram-1);
  cylinderPoint[1] = radius*(yOnPolarDiagram);
  cylinderPoint[2] = yOnGrid;
  cylinderCentre[0] = radius;
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

  if ( outputTrackingModel.length() == 0
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
  double blockSizeInMillimetres = actualTagSizeIncludingBorder/static_cast<double>(9);
  double borderSizeInMillimetres = actualTagSizeIncludingBorder / static_cast<double>(9);
  double cornerOffsetInMillimetres = blockSizeInMillimetres * 7;
  double centreOffsetInMillimetres = blockSizeInMillimetres * 3.5;
  double spacingBetweenTags = blockSizeInMillimetres * 2;

  double printerDotsPerMillimetre = (printerDotsPerInch / mmPerInch);
  int numberOfAvailablePixelsAlongLength = (int)(length*printerDotsPerMillimetre);
  int numberOfTagsAlongLength = std::floor(static_cast<double>(length)/static_cast<double>(actualTagSizeIncludingBorder));
  int numberOfPixelsRequiredAlongLength = numberOfTagsAlongLength*inputTagSize;

  double maxCircumferenceInMillimetres = pi * diameter * ((pi - asin(surface/diameter))/pi); // distance around probe; ie. section of circumference
  int numberTagsAlongWidth = std::floor(static_cast<double>(maxCircumferenceInMillimetres)/static_cast<double>(actualTagSizeIncludingBorder));
  int numberOfPixelsRequiredAlongWidth = numberTagsAlongWidth*inputTagSize;

  double minXInMillimetres = 0 - borderSizeInMillimetres;
  double minYInMillimetres = 0 - borderSizeInMillimetres;
  double maxXInMillimetres = minXInMillimetres + (numberTagsAlongWidth*actualTagSizeIncludingBorder);
  double maxYInMillimetres = minYInMillimetres + (numberOfTagsAlongLength*actualTagSizeIncludingBorder);

  double cyclinderLength = numberOfTagsAlongLength*actualTagSizeIncludingBorder;

  std::cout << "dots per inch to print at      = " << printerDotsPerInch << std::endl;
  std::cout << "dots per millimetres           = " << printerDotsPerMillimetre << std::endl;
  std::cout << "number of pixels available     = " << numberOfAvailablePixelsAlongLength << std::endl;
  std::cout << "number of tags along length    = " << numberOfTagsAlongLength << std::endl;
  std::cout << "circumference in millimetres   = " << maxCircumferenceInMillimetres << std::endl;
  std::cout << "number of tags along width     = " << numberTagsAlongWidth << std::endl;
  std::cout << "block size in millimetres      = " << blockSizeInMillimetres << std::endl;
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
  // 5. Origin is the corner of the first tag.... NOT including the white border.

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

      ConvertGridPointToCyclinderPoint(pointID+0,     lengthCounter, widthCounter, centreOffsetInMillimetres, centreOffsetInMillimetres, actualTagSizeIncludingBorder, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+10000, lengthCounter, widthCounter, 0,                         0,                         actualTagSizeIncludingBorder, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+20000, lengthCounter, widthCounter, cornerOffsetInMillimetres, 0,                         actualTagSizeIncludingBorder, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+30000, lengthCounter, widthCounter, cornerOffsetInMillimetres, cornerOffsetInMillimetres, actualTagSizeIncludingBorder, radius, points, normals, pointIDArray, vertices);
      ConvertGridPointToCyclinderPoint(pointID+40000, lengthCounter, widthCounter, 0,                         cornerOffsetInMillimetres, actualTagSizeIncludingBorder, radius, points, normals, pointIDArray, vertices);
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
    vtkSmartPointer<vtkCylinderSource> cylinderSource = vtkCylinderSource::New();
    cylinderSource->SetCenter(0, cyclinderLength/2.0 - borderSizeInMillimetres, 0);
    cylinderSource->SetRadius(radius);
    cylinderSource->SetHeight(cyclinderLength);
    cylinderSource->SetResolution(360);
    cylinderSource->SetCapping(false);

    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkTriangleFilter::New();
    triangleFilter->SetInputConnection(cylinderSource->GetOutputPort());

    vtkSmartPointer<vtkLinearSubdivisionFilter> subdivide = vtkLinearSubdivisionFilter::New();
    subdivide->SetInputConnection(triangleFilter->GetOutputPort());
    subdivide->SetNumberOfSubdivisions(7);

    vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
    transform->Identity();
    transform->RotateX(90);
    transform->Translate(radius, 0, 0);

    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkTransformPolyDataFilter::New();
    transformFilter->SetInput(subdivide->GetOutput());
    transformFilter->SetTransform(transform);
    transformFilter->Update();

    vtkSmartPointer<vtkDoubleArray> normals2 = vtkDoubleArray::New();
    normals2->SetNumberOfComponents(3);
    normals2->SetName("Normals");
    normals2->Initialize();

    if (textureMap.size() > 0)
    {
      vtkPoints *points = transformFilter->GetOutput()->GetPoints();
      vtkIdType numberPoints = points->GetNumberOfPoints();

      vtkFloatArray* tc = vtkFloatArray::New();
      tc->SetNumberOfComponents( 2 );
      tc->Allocate(numberPoints);

      for (vtkIdType counter = 0; counter < numberPoints; counter++)
      {        
        double *p = points->GetPoint(counter);

        // Need to measure distance around curvature of probe.
        double norm[2] = {0, 0};
        norm[0] = p[0] - radius;
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

        if (dx > ((numberTagsAlongWidth*actualTagSizeIncludingBorder) - borderSizeInMillimetres)
            )
        {
          dx = 1*borderSizeInMillimetres;
          dy = 1*borderSizeInMillimetres;
        }

        double tx = (dx-minXInMillimetres) / (maxXInMillimetres - minXInMillimetres);
        double ty = -(dy-minYInMillimetres) / (maxYInMillimetres - minYInMillimetres);

        // Convert to texture coord.
        tc->InsertNextTuple2(tx, ty);
        normals2->InsertNextTuple3(norm[0], norm[1], 0);
      }
      transformFilter->GetOutput()->GetPointData()->SetTCoords(tc);
      transformFilter->GetOutput()->GetPointData()->SetNormals(normals2);
    }

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();
    writer->SetInput(transformFilter->GetOutput());
    writer->SetFileName(outputVisualisationModel.c_str());
    writer->Update();

    std::cout << "written visualisation model to = " << outputVisualisationModel << std::endl;
  }

  return EXIT_SUCCESS;
}
