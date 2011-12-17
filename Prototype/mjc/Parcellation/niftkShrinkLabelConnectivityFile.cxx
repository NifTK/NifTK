/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkType.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include <limits>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a label connectivity file, and two lower resolution labelled spheres," << std::endl;
    std::cout << "  and outputs low resolution connectivity file, with as many tracks attached as possible." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -con connectivity.vtk -lhs lh.sphere.vtk -rhs rh.sphere.vtk -o output.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -con  <filename>          Input VTK Poly Data file containing connectivity lines in mm space, in space of T1 image." << std::endl;
    std::cout << "    -lhs  <filename>          Input VTK Poly Data file, left hand low resolution spherical surface, including labels." << std::endl;
    std::cout << "    -rhs  <filename>          Input VTK Poly Data file, right hand low resolution spherical surface, including labels." << std::endl;
    std::cout << "    -o    <filename>          Output VTK Poly Data file containing spherical surfaces, connectivity lines and labels." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -d <float> [2]            Distance threshold. If track is further than this, then point is not connected to the track." << std::endl << std::endl;
    
  }

struct arguments
{
  std::string inputConnectivityFile;
  std::string outputConnectivityFile;
  std::string inputLeftSideSphericalFile;
  std::string inputRightSideSphericalFile;
  double distanceThreshold;  
};


double distanceBetweenPoints(double* a, double *b)
  {
    return sqrt(
          ((a[0]-b[0]) * (a[0]-b[0])) 
        + ((a[1]-b[1]) * (a[1]-b[1]))
        + ((a[2]-b[2]) * (a[2]-b[2]))
        );
  }

vtkIdType getClosestPointOnOneHemisphere(
    vtkPoints *points,
    double *point
    )
  {

    double distance = 0;
    vtkIdType closestIndex = -1;
    double closestDistance = std::numeric_limits<double>::max();
    double pointOnSurface[3];
    
    vtkIdType numberOfPoints = points->GetNumberOfPoints();
    
    for (vtkIdType i = 0; i < numberOfPoints; i++)
      {
        points->GetPoint(i, pointOnSurface);
        
        distance =  distanceBetweenPoints(point, pointOnSurface);
        if (distance < closestDistance)
          {
            closestIndex = i;
            closestDistance = distance;
          }
      }
    
    return closestIndex;
  }

void getClosestPointOnEitherHemisphere(
    vtkPoints *leftSurfacePoints, 
    vtkPoints *rightSurfacePoints,
    double *point,
    vtkIdType& outputIndex,
    bool& outputIsLeftHemisphere
    )
  {
    double leftPoint[3];
    double rightPoint[3];
    
    vtkIdType leftIndex = getClosestPointOnOneHemisphere(leftSurfacePoints, point);
    vtkIdType rightIndex = getClosestPointOnOneHemisphere(rightSurfacePoints, point);
    
    leftSurfacePoints->GetPoint(leftIndex, leftPoint);
    rightSurfacePoints->GetPoint(rightIndex, rightPoint);

    double leftDistance = distanceBetweenPoints(point, leftPoint);
    double rightDistance = distanceBetweenPoints(point, rightPoint);
    
    /*
    std::cout << "Point=" << point[0] << ", " << point[1] << ", " << point[2] << ", Left closest = " << leftPoint[0] << ", " << leftPoint[1] << ", " << leftPoint[2] << ", di=" << leftDistance << std::endl;
    std::cout << "Point=" << point[0] << ", " << point[1] << ", " << point[2] << ", Right closest = " << rightPoint[0] << ", " << rightPoint[1] << ", " << rightPoint[2] << ", di=" << rightDistance << std::endl;
    */
    
    outputIndex = -1;
    
    if (leftDistance < rightDistance)
      {
        outputIndex = leftIndex;
        outputIsLeftHemisphere = true;
      }
    else
      {
        outputIndex = rightIndex;
        outputIsLeftHemisphere = false;
      }
  }

void createConnections(
    vtkPoints *leftSurfacePoints,
    vtkPoints *rightSurfacePoints,
    vtkPoints *connectivityPoints,
    vtkPolyData *connectivityPolyData,
    double threshold,
    vtkCellArray *outputLines
    )
  {
    double startPoint[3];
    double endPoint[3];
    vtkIdType startIndex;
    vtkIdType endIndex;
    bool startIndexIsInLeftHemisphere;
    bool endIndexIsInLeftHemisphere;
    double closestToStartPoint[3];
    double closestToEndPoint[3];
    vtkIdType *outputLine = new vtkIdType[2];
    vtkIdType *pointsInCell = new vtkIdType[2];
    vtkIdType numberOfPointsInCell;
    
    vtkCellArray *lines = connectivityPolyData->GetLines();
    
    unsigned long int counter = 0;
    lines->InitTraversal();
    
    while(lines->GetNextCell(numberOfPointsInCell, pointsInCell))
      {
        counter++;
        
        if (numberOfPointsInCell != 2)
          {
            std::cerr << "Cell number " << counter << ", is not a line, we have " << numberOfPointsInCell << std::endl;
          }
        else
          {
            connectivityPoints->GetPoint(pointsInCell[0], startPoint);
            connectivityPoints->GetPoint(pointsInCell[1], endPoint);
            
            getClosestPointOnEitherHemisphere(leftSurfacePoints, rightSurfacePoints, startPoint, startIndex, startIndexIsInLeftHemisphere);
            getClosestPointOnEitherHemisphere(leftSurfacePoints, rightSurfacePoints, endPoint, endIndex, endIndexIsInLeftHemisphere);
            
            if (startIndex == -1) 
              {
                std::cerr << "Cell number " << counter << ", index 0, failed to find closest point" << std::endl;
              }
            
            if (endIndex == -1) 
              {
                std::cerr << "Cell number " << counter << ", index 1, failed to find closest point" << std::endl;
              }
            
            if (startIndex != -1 && endIndex != -1)
              {
               
                if (startIndexIsInLeftHemisphere)
                  {
                    leftSurfacePoints->GetPoint(startIndex, closestToStartPoint);
                  }
                else
                  {
                    rightSurfacePoints->GetPoint(startIndex, closestToStartPoint);
                  }

                if (endIndexIsInLeftHemisphere)
                  {
                    leftSurfacePoints->GetPoint(endIndex, closestToEndPoint);
                  }
                else
                  {
                    rightSurfacePoints->GetPoint(endIndex, closestToEndPoint);
                  }

                if (distanceBetweenPoints(startPoint, closestToStartPoint) < threshold && 
                    distanceBetweenPoints(endPoint, closestToEndPoint) < threshold)
                  {

                    if (!startIndexIsInLeftHemisphere)
                      {
                        startIndex += leftSurfacePoints->GetNumberOfPoints();
                      }
                    
                    if (!endIndexIsInLeftHemisphere)
                      {
                        endIndex += leftSurfacePoints->GetNumberOfPoints();
                      }
                    outputLine[0] = startIndex;
                    outputLine[1] = endIndex;
                    
                    outputLines->InsertNextCell(2, outputLine);

                    if (outputLines->GetNumberOfCells() % 1000 == 0)
                      {
                        std::cout << "#lines added=" << outputLines->GetNumberOfCells() << std::endl;
                      }                    
                  }
                else
                  {
                    std::cerr << "Cell number " << counter << ", has distance " << distanceBetweenPoints(startPoint, closestToStartPoint) \
                      << ", " << distanceBetweenPoints(endPoint, closestToEndPoint) \
                      << ", so is above threshold " << threshold \
                      << std::endl;
                  }
              }
          }
      }    
  }

/**
 * \brief Combines various VTK poly data file into one connectivity file.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.distanceThreshold = 2;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-con") == 0){
      args.inputConnectivityFile=argv[++i];
      std::cout << "Set -con=" << args.inputConnectivityFile;
    }
    else if(strcmp(argv[i], "-lhs") == 0){
      args.inputLeftSideSphericalFile=argv[++i];
      std::cout << "Set -lhs=" << args.inputLeftSideSphericalFile;
    }
    else if(strcmp(argv[i], "-rhs") == 0){
      args.inputRightSideSphericalFile=argv[++i];
      std::cout << "Set -rhs=" << args.inputRightSideSphericalFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputConnectivityFile=argv[++i];
      std::cout << "Set -o=" << args.outputConnectivityFile;
    }
    else if(strcmp(argv[i], "-d") == 0){
      args.distanceThreshold=atof(argv[++i]);
      std::cout << "Set -d=" << niftk::ConvertToString(args.distanceThreshold);
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.outputConnectivityFile.length() == 0 || 
      args.inputConnectivityFile.length() == 0 ||
      args.inputLeftSideSphericalFile.length() == 0 ||
      args.inputRightSideSphericalFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  try
  {
    // Load datasets
    vtkPolyDataReader *connectivityReader = vtkPolyDataReader::New();
    connectivityReader->SetFileName(args.inputConnectivityFile.c_str());
    connectivityReader->Update();
    
    vtkPolyDataReader *lhSphereReader = vtkPolyDataReader::New();
    lhSphereReader->SetFileName(args.inputLeftSideSphericalFile.c_str());
    lhSphereReader->Update();
    
    vtkPolyDataReader *rhSphereReader = vtkPolyDataReader::New();
    rhSphereReader->SetFileName(args.inputRightSideSphericalFile.c_str());
    rhSphereReader->Update();
    
    std::cout << "Input connectivity file:" << std::endl;
    std::cout << "\tPoints:" << connectivityReader->GetOutput()->GetNumberOfPoints() << std::endl;
    std::cout << "\tPolys:" << connectivityReader->GetOutput()->GetPolys()->GetNumberOfCells() << std::endl;
    std::cout << "\tLines:" << connectivityReader->GetOutput()->GetLines()->GetNumberOfCells() << std::endl;
    
    unsigned long int numberOfPointsInLeftHemi = lhSphereReader->GetOutput()->GetNumberOfPoints();
    unsigned long int numberOfPointsInRightHemi = rhSphereReader->GetOutput()->GetNumberOfPoints();

    unsigned long int numberOfCellsInLeftHemi = lhSphereReader->GetOutput()->GetPolys()->GetNumberOfCells();
    unsigned long int numberOfCellsInRightHemi = rhSphereReader->GetOutput()->GetPolys()->GetNumberOfCells();
    
    std::cout << "Left sphere:" << std::endl;
    std::cout << "\tPoints:" << numberOfPointsInLeftHemi << std::endl;
    std::cout << "\tCells:" << numberOfCellsInLeftHemi << std::endl;

    std::cout << "Right sphere:" << std::endl;
    std::cout << "\tPoints:" << numberOfPointsInRightHemi << std::endl;
    std::cout << "\tCells:" << numberOfCellsInRightHemi << std::endl;

    // First we tranlsate the two input sphere's along x, y, z axis.
    // SEE ALSO: niftkMergeConnectivityWithSpheres
    
    vtkMatrix4x4 *leftMatrix = vtkMatrix4x4::New();
    leftMatrix->SetElement(0,3,-100);
    leftMatrix->SetElement(1,3,-100);
    leftMatrix->SetElement(2,3,-100);
    
    vtkTransform *leftTransform = vtkTransform::New();
    leftTransform->Identity();
    leftTransform->PostMultiply();
    leftTransform->Concatenate(leftMatrix);
    
    vtkTransformPolyDataFilter *leftFilter =  vtkTransformPolyDataFilter::New();
    leftFilter->SetInput(lhSphereReader->GetOutput());
    leftFilter->SetTransform(leftTransform);
    leftFilter->Update();
    
    vtkMatrix4x4 *rightMatrix = vtkMatrix4x4::New();
    rightMatrix->SetElement(0,3,+100);
    rightMatrix->SetElement(1,3,+100);
    rightMatrix->SetElement(2,3,+100);
    
    vtkTransform *rightTransform = vtkTransform::New();
    rightTransform->Identity();
    rightTransform->PostMultiply();
    rightTransform->Concatenate(rightMatrix);
    
    vtkTransformPolyDataFilter *rightFilter =  vtkTransformPolyDataFilter::New();
    rightFilter->SetInput(rhSphereReader->GetOutput());
    rightFilter->SetTransform(rightTransform);
    rightFilter->Update();
    
    // Second we merge the two spheres

    vtkPoints *leftHemiPoints = leftFilter->GetOutput()->GetPoints();
    vtkPoints *rightHemiPoints = rightFilter->GetOutput()->GetPoints();
    
    vtkIntArray *leftHemiLabels = dynamic_cast<vtkIntArray*>(leftFilter->GetOutput()->GetPointData()->GetScalars());
    vtkIntArray *rightHemiLabels = dynamic_cast<vtkIntArray*>(rightFilter->GetOutput()->GetPointData()->GetScalars());
    
    vtkCellArray *leftCellArray = leftFilter->GetOutput()->GetPolys();
    vtkCellArray *rightCellArray = rightFilter->GetOutput()->GetPolys();
    
    vtkPoints *outputPoints = vtkPoints::New();
    outputPoints->SetDataTypeToFloat();
    outputPoints->Allocate(numberOfPointsInLeftHemi + numberOfPointsInRightHemi);
    
    vtkIntArray *outputLabels = vtkIntArray::New();
    outputLabels->SetNumberOfComponents(1);
    outputLabels->SetNumberOfValues(numberOfPointsInLeftHemi + numberOfPointsInRightHemi);
    
    vtkCellArray *outputTriangles = vtkCellArray::New();
    vtkCellArray *outputLines = vtkCellArray::New();
    
    double point[3];
    vtkIdType *triangle = new vtkIdType[3];
    vtkIdType numberOfPoints;
        
    for (unsigned long int i = 0; i < numberOfPointsInLeftHemi; i++)
      {
        leftHemiPoints->GetPoint(i, point);
        outputPoints->InsertPoint(i, point);
        outputLabels->SetValue(i, (int)leftHemiLabels->GetTuple1(i));
      }
    
    for (unsigned long int i = 0; i < numberOfPointsInRightHemi; i++)
      {
        rightHemiPoints->GetPoint(i, point);
        outputPoints->InsertPoint(i+numberOfPointsInLeftHemi, point);
        outputLabels->SetValue(i+numberOfPointsInLeftHemi, (int)rightHemiLabels->GetTuple1(i));        
      }
    
    leftCellArray->InitTraversal();
    for (unsigned long int i = 0; i < numberOfCellsInLeftHemi; i++)
      {
        
        leftCellArray->GetNextCell(numberOfPoints, triangle);
        outputTriangles->InsertNextCell(numberOfPoints, triangle);
        
      }
    
    rightCellArray->InitTraversal();
    for (unsigned long int i = 0; i < numberOfCellsInRightHemi; i++)
      {
        
        rightCellArray->GetNextCell(numberOfPoints, triangle);
        
        for (int j = 0; j < numberOfPoints; j++)
          {
            triangle[j] += numberOfPointsInLeftHemi;
          }
        outputTriangles->InsertNextCell(numberOfPoints, triangle);
      }

    vtkPolyData *connectivityPolyData = connectivityReader->GetOutput();
    connectivityPolyData->BuildCells();
    connectivityPolyData->BuildLinks();

    vtkPoints *leftSpherePoints = leftFilter->GetOutput()->GetPoints();
    vtkPoints *rightSpherePoints = rightFilter->GetOutput()->GetPoints();
    vtkPoints *connectivityPoints = connectivityReader->GetOutput()->GetPoints();

    createConnections(
        leftSpherePoints,
        rightSpherePoints,
        connectivityPoints,
        connectivityPolyData,
        args.distanceThreshold,
        outputLines);

    std::cout << "Output connectivity file:" << std::endl;
    std::cout << "\tPoints:" << outputPoints->GetNumberOfPoints() << std::endl;
    std::cout << "\tPolys:" << outputTriangles->GetNumberOfCells() << std::endl;
    std::cout << "\tLines:" << outputLines->GetNumberOfCells() << std::endl;
  
    vtkPolyData *outputPolyData = vtkPolyData::New();
    outputPolyData->SetPoints(outputPoints);
    outputPolyData->GetPointData()->SetScalars(outputLabels);
    outputPolyData->SetPolys(outputTriangles);
    outputPolyData->SetLines(outputLines);

    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
    writer->SetFileName(args.outputConnectivityFile.c_str());
    writer->SetInput(outputPolyData);
    writer->SetFileTypeToASCII();
    writer->Update();
    
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
  return EXIT_SUCCESS;

}
