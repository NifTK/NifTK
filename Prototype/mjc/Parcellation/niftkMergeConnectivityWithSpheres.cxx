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
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkType.h"
#include <limits>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a tracktography connectivity file, and left+right, white matter + spherical surfaces" << std::endl;
    std::cout << "  and outputs a single Poly Data file which is a surface with labels and connectivity." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -con connectivity.vtk -lhw lh.white.vtk -rhw rh.white.vtk -lhs lh.sphere.vtk -rhs rh.sphere.vtk -o output.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -con  <filename>          Input VTK Poly Data file containing connectivity lines in mm space, in space of T1 image." << std::endl;
    std::cout << "    -lhw  <filename>          Input VTK Poly Data file, left side white matter surface, in space of T1 image." << std::endl;      
    std::cout << "    -rhw  <filename>          Input VTK Poly Data file, right side white matter surface, in space of T1 image." << std::endl;
    std::cout << "    -lhs  <filename>          Input VTK Poly Data file, left hand spherical surface, including labels, as produced by niftk_freesurfer_annotation_to_vtk.m." << std::endl;
    std::cout << "    -rhs  <filename>          Input VTK Poly Data file, right hand spherical surface, including labels, as produced by niftk_freesurfer_annotation_to_vtk.m." << std::endl;
    std::cout << "    -o    <filename>          Output VTK Poly Data file containing spherical surfaces, connectivity lines and labels." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -d <float> [2]            Distance threshold. If closest track is further than this, then point is not connected to the track." << std::endl;
    std::cout << "    -pointBased               Normally we say 'for each track, find closest point', with -pointBased we say 'for each point, find closest track'" << std::endl << std::endl;
    std::cout << "    -maxLines [int]           Only do the first n lines (useful for preparing images for papers)" << std::endl;
    std::cout << "    -noOutputTriangles        Don't output triangles on output (useful for preparing images for papers)" << std::endl;
  }

struct arguments
{
  std::string inputConnectivityFile;
  std::string inputLeftSideWhiteMatterFile;
  std::string inputRightSideWhiteMatterFile;
  std::string inputLeftSideSphericalFile;
  std::string inputRightSideSphericalFile;
  std::string outputPolyDataFile;
  double distanceThreshold;
  bool usePointBased;
  unsigned long int maxLines;
  bool outputTriangles;
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
    
    vtkIdType closestIndex = -1;
    
    double distance = 0;
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

bool getClosestPointOnEitherHemisphere(
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
    
    if (leftIndex == -1)
      {
        std::cerr << "ERROR: Point[" << point[0] << ", " << point[1] << ", " << point[2] << " didnt have closes point on left hemi" << std::endl;
      }

    if (rightIndex == -1)
      {
        std::cerr << "ERROR: Point[" << point[0] << ", " << point[1] << ", " << point[2] << " didnt have closes point on right hemi" << std::endl;
      }

    if (leftIndex == -1 || rightIndex == -1)
      {
        return false;
      }
    
    leftSurfacePoints->GetPoint(leftIndex, leftPoint);
    rightSurfacePoints->GetPoint(rightIndex, rightPoint);
    
    outputIndex = -1;
    
    if (distanceBetweenPoints(point, leftPoint) < distanceBetweenPoints(point, rightPoint))
      {
        outputIndex = leftIndex;
        outputIsLeftHemisphere = true;
      }
    else
      {
        outputIndex = rightIndex;
        outputIsLeftHemisphere = false;
      }
    return true;
  }

void createConnections2(
    vtkPoints *leftSurfacePoints,
    vtkPoints *rightSurfacePoints,
    vtkPoints *connectivityPoints,
    vtkPolyData *connectivityPolyData,
    double threshold,
    vtkIdType maxNumberLines,
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
    vtkIdType numberOfPointsInLeftSurface = leftSurfacePoints->GetNumberOfPoints();
    vtkIdType numberOfPointsInRightSurface = rightSurfacePoints->GetNumberOfPoints();
    vtkIdType totalNumberOfPoints = numberOfPointsInLeftSurface + numberOfPointsInRightSurface;
    
    vtkCellArray *lines = connectivityPolyData->GetLines();
    
    long int counter = 0;
    lines->InitTraversal();
    
    while(lines->GetNextCell(numberOfPointsInCell, pointsInCell) && counter < maxNumberLines)
      {
        counter++;
        
        if (numberOfPointsInCell != 2)
          {
            std::cerr << "ERROR: Cell number " << counter << ", is not a line, we have " << numberOfPointsInCell << std::endl;
          }
        else
          {
            connectivityPoints->GetPoint(pointsInCell[0], startPoint);
            connectivityPoints->GetPoint(pointsInCell[1], endPoint);
            
            bool foundStart = getClosestPointOnEitherHemisphere(leftSurfacePoints, rightSurfacePoints, startPoint, startIndex, startIndexIsInLeftHemisphere);
            bool foundEnd = getClosestPointOnEitherHemisphere(leftSurfacePoints, rightSurfacePoints, endPoint, endIndex, endIndexIsInLeftHemisphere);
            
            if (!foundStart) 
              {
                std::cerr << "ERROR: Cell number " << counter << ", failed to find closest point to start" << std::endl;
              }
            
            if (!foundEnd) 
              {
                std::cerr << "ERROR: Cell number " << counter << ", failed to find closest point to end" << std::endl;
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

                for (unsigned int i = 0; i < 3; i++)
                  {
                    if (closestToStartPoint[i] < -200 || closestToStartPoint[i] > 200)
                      {
                        std::cerr << "ERROR: Start point " << counter << ", index " << i << ", is out of range" << std::endl;
                      }
                    if (closestToEndPoint[i] < -200 || closestToEndPoint[i] > 200)
                      {
                        std::cerr << "ERROR: End point " << counter << ", index " << i << ", is out of range" << std::endl;
                      }
                    
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
                    
                    if (startIndex < 0 || startIndex > totalNumberOfPoints)
                      {
                        std::cerr << "ERROR: Start point " << counter << " has an out of range index" << std::endl;
                      }

                    if (endIndex < 0 || endIndex > totalNumberOfPoints)
                      {
                        std::cerr << "ERROR: End point " << counter << " has an out of range index" << std::endl;
                      }

                    outputLine[0] = startIndex;
                    outputLine[1] = endIndex;
                    
                    outputLines->InsertNextCell(2, outputLine);

                    if (outputLines->GetNumberOfCells() % 100 == 0)
                      {
                        std::cout << "#cells added=" << outputLines->GetNumberOfCells() << std::endl;
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
            else
              {
                std::cerr << "Skipping line " << counter << ", as we failed to find start or end point" << std::endl;
              }
          }
      }
  }

void createConnections(
    bool isLeft,
    unsigned long int numberOfPointsInThisHemi,
    unsigned long int numberOfPointsInOtherHemi,
    vtkPoints *thisWhiteMatterPoints,
    vtkPoints *otherWhiteMatterPoints,
    vtkPoints *connectivityPoints,
    vtkPolyData *connectivityPolyData,
    double threshold,
    vtkIdType maxNumberLines,
    vtkCellArray *outputLines
    )
  {
    double point[3];
    double connectivityPoint[3];
    double oppositePoint[3];
    double oppositePointsClosestPoint[3];
    short unsigned int numberOfCells;
    vtkIdType numberOfPointsInCurrentCell;
    vtkIdType *line = new vtkIdType[2];
    vtkIdType *outputLine = new vtkIdType[2];
    vtkIdType *cellIds = new vtkIdType[100];
    
    unsigned long int numberOfPointsInConnectivityFile =  connectivityPoints->GetNumberOfPoints();
    unsigned long int numberConnectedToThisHemisphere = 0;
    unsigned long int numberConnectedToOtherHemisphere = 0;
    
    for (unsigned long int i = 0; i < numberOfPointsInThisHemi; i++)
      {
        // FreeSurfer outputs in RAS coordinates.
        // However, we assume that data is all in T1 mm space.
        // i.e. you have corrected for, and checked (using Paraview say) the transformed data beforehand.
        
        thisWhiteMatterPoints->GetPoint(i, point);
        
        // Find closes point in connectivity volume.
        double distance = 0;
        double bestDistance = std::numeric_limits<double>::max();
        
        vtkIdType bestIndex = -1;
        vtkIdType bestIndexForOppositeEnd = -1;
        
        for (unsigned long int j = 0; j < numberOfPointsInConnectivityFile; j++)
          {
            connectivityPoints->GetPoint(j, connectivityPoint);
            connectivityPolyData->GetPointCells(j, numberOfCells, cellIds);
            
            distance = distanceBetweenPoints(point, connectivityPoint);
            
            if (distance < bestDistance && numberOfCells > 0)
              {
                bestDistance = distance;
                bestIndex = j;
              }
          }
        
        if (bestDistance < threshold)
          {
            // Find all cells containing this point. i.e. many connectivity lines could be connected to point
            connectivityPolyData->GetPointCells(bestIndex, numberOfCells, cellIds);
            
            if (numberOfCells > 0)
              {
                // For each cell, find the other end of the cell, and the point location thereof.
                for (short unsigned k = 0; k < numberOfCells; k++)
                  {
                    connectivityPolyData->GetCellPoints(cellIds[k], numberOfPointsInCurrentCell, line);

                    // These should all be lines.
                    if (numberOfPointsInCurrentCell != 2)
                      {
                        std::cerr << "i=" << i << ", Warning, point number " << bestIndex << ", in cell " << cellIds[k] << ", is in a cell with " << numberOfPointsInCurrentCell << " points " << std::endl;
                      }
                    else
                      {
                        for (int l = 0; l < numberOfPointsInCurrentCell; l++)
                          {
                            
                            if (line[l] != bestIndex)
                              {
                                connectivityPoints->GetPoint(line[l], oppositePoint);
                                
                                // Now we need to seach both hemispheres to find closest point.
                                
                                double distanceToThis = 0;
                                double bestDistanceForOppositeEnd = std::numeric_limits<double>::max();
                                bool bestIndexInThisHemisphere = false;
                                
                                for (unsigned long int x = 0; x < numberOfPointsInThisHemi; x++)
                                  {
                                    thisWhiteMatterPoints->GetPoint(x, oppositePointsClosestPoint);
                                    
                                    distanceToThis = distanceBetweenPoints(oppositePoint, oppositePointsClosestPoint);
                                    
                                    if (distanceToThis < bestDistanceForOppositeEnd)
                                      {
                                        bestDistanceForOppositeEnd = distanceToThis;
                                        bestIndexForOppositeEnd = x;
                                        bestIndexInThisHemisphere = true;
                                      }
                                  }
                                
                                double distanceToOther = 0;
                                
                                for (unsigned long int x = 0; x < numberOfPointsInOtherHemi; x++)
                                  {
                                    otherWhiteMatterPoints->GetPoint(x, oppositePointsClosestPoint);
                                    
                                    distanceToOther = distanceBetweenPoints(oppositePoint, oppositePointsClosestPoint);
                                    
                                    if (distanceToOther < bestDistanceForOppositeEnd)
                                      {
                                        bestDistanceForOppositeEnd = distanceToOther;
                                        bestIndexForOppositeEnd = x;
                                        bestIndexInThisHemisphere = false;
                                      }
                                  }
                                
                                if (bestDistanceForOppositeEnd < threshold)
                                  {
 
                                    unsigned long int firstIndex = i;
                                    unsigned long int secondIndex = bestIndexForOppositeEnd;
                                    
                                    if (bestIndexInThisHemisphere)
                                      {
                                        numberConnectedToThisHemisphere++;
                                      }
                                    else
                                      {
                                        numberConnectedToOtherHemisphere++;                                    
                                      }

                                    if (isLeft)
                                      {
                                        if (!bestIndexInThisHemisphere)
                                          {
                                            secondIndex += numberOfPointsInThisHemi;    
                                          }
                                      }
                                    else
                                      {
                                        firstIndex += numberOfPointsInOtherHemi;
                                        
                                        if (bestIndexInThisHemisphere)
                                          {
                                            secondIndex += numberOfPointsInOtherHemi;
                                          }
                                      }

                                    // Now, create a cell.
                                    outputLine[0] = firstIndex;
                                    outputLine[1] = secondIndex;
                                    outputLines->InsertNextCell(2, outputLine);

                                    if (outputLines->GetNumberOfCells() % 100 == 0)
                                      {
                                        std::cout << "left=" << isLeft << ", total=" << outputLines->GetNumberOfCells() \
                                          << " cells, thisSide=" << numberConnectedToThisHemisphere \
                                          << " t'otherSide=" << numberConnectedToOtherHemisphere \
                                          << std::endl;
                                        
                                      }
                                    
                                    if (outputLines->GetNumberOfCells() >= maxNumberLines)
                                      {
                                        break;
                                      }
                                  }
/*
                                else
                                  {
                                    if (!isLeft)
                                      {
                                        std::cout << "Didn't find point at other end of line." << std::endl;
                                      }
                                  }
*/                                  
                              } // end if we found the other end of the line
                          } // end for each point in line
                      } // end if we are processing a line
                  } // end for each cell containing point                
              }
/*            
            else
              {
                if (!isLeft)
                  {
                    std::cout << "Point found is in no cells" << std::endl;
                  }
              }
*/              
          } // end if we are below threshold
/*        
        else
          {
            if (!isLeft)
              {
                std::cout << "Failed to find line for point " << i << std::endl;
              }
          }
*/          
      } // for each point in left hemmi.    
    
  }


/**
 * \brief Combines various VTK poly data file into one connectivity file.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.distanceThreshold = 2;
  args.usePointBased = false;
  args.maxLines = -1;
  args.outputTriangles = true;
  
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
    else if(strcmp(argv[i], "-lhw") == 0){
      args.inputLeftSideWhiteMatterFile=argv[++i];
      std::cout << "Set -lhw=" << args.inputLeftSideWhiteMatterFile;
    }
    else if(strcmp(argv[i], "-rhw") == 0){
      args.inputRightSideWhiteMatterFile=argv[++i];
      std::cout << "Set -rhw=" << args.inputRightSideWhiteMatterFile;
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
      args.outputPolyDataFile=argv[++i];
      std::cout << "Set -o=" << args.outputPolyDataFile;
    }
    else if(strcmp(argv[i], "-d") == 0){
      args.distanceThreshold=atof(argv[++i]);
      std::cout << "Set -d=" + niftk::ConvertToString(args.distanceThreshold);
    }    
    else if(strcmp(argv[i], "-pointBased") == 0){
      args.usePointBased=true;
      std::cout << "Set -pointBased=" << niftk::ConvertToString(args.usePointBased);
    }
    else if(strcmp(argv[i], "-maxLines") == 0){
      args.maxLines=atoi(argv[++i]);
      std::cout << "Set -maxLines=" << niftk::ConvertToString((int)args.maxLines);
    }
    else if(strcmp(argv[i], "-noOutputTriangles") == 0){
      args.outputTriangles=false;
      std::cout << "Set -noOutputTriangles=" << niftk::ConvertToString(args.outputTriangles);
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.outputPolyDataFile.length() == 0 || 
      args.inputConnectivityFile.length() == 0 ||
      args.inputLeftSideSphericalFile.length() == 0 ||
      args.inputLeftSideWhiteMatterFile.length() == 0 ||
      args.inputRightSideSphericalFile.length() == 0 ||
      args.inputRightSideWhiteMatterFile.length() == 0)
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
    
    vtkPolyDataReader *lhWhiteReader = vtkPolyDataReader::New();
    lhWhiteReader->SetFileName(args.inputLeftSideWhiteMatterFile.c_str());
    lhWhiteReader->Update();
    
    vtkPolyDataReader *rhWhiteReader = vtkPolyDataReader::New();
    rhWhiteReader->SetFileName(args.inputRightSideWhiteMatterFile.c_str());
    rhWhiteReader->Update();
    
    vtkPolyDataReader *lhSphereReader = vtkPolyDataReader::New();
    lhSphereReader->SetFileName(args.inputLeftSideSphericalFile.c_str());
    lhSphereReader->Update();
    
    vtkPolyDataReader *rhSphereReader = vtkPolyDataReader::New();
    rhSphereReader->SetFileName(args.inputRightSideSphericalFile.c_str());
    rhSphereReader->Update();
    
    std::cout << "Points in connectivity=" << connectivityReader->GetOutput()->GetNumberOfPoints() << std::endl;
    std::cout << "Polys in connectivity=" << connectivityReader->GetOutput()->GetPolys()->GetNumberOfCells() << std::endl;
    std::cout << "Lines in connectivity=" << connectivityReader->GetOutput()->GetLines()->GetNumberOfCells() << std::endl;
    
    unsigned long int numberOfPointsInLeftHemi = lhSphereReader->GetOutput()->GetNumberOfPoints();
    unsigned long int numberOfPointsInRightHemi = rhSphereReader->GetOutput()->GetNumberOfPoints();
    
    std::cout << "Points in left=" << numberOfPointsInLeftHemi << ", points in right=" << numberOfPointsInRightHemi << std::endl;
    
    unsigned long int numberOfCellsInLeftHemi = lhSphereReader->GetOutput()->GetPolys()->GetNumberOfCells();
    unsigned long int numberOfCellsInRightHemi = rhSphereReader->GetOutput()->GetPolys()->GetNumberOfCells();
    
    std::cout << "Cells in left=" << numberOfCellsInLeftHemi << ", cells in right=" << numberOfCellsInRightHemi << std::endl;
    
    vtkPoints *leftHemiPoints = lhSphereReader->GetOutput()->GetPoints();
    vtkPoints *rightHemiPoints = rhSphereReader->GetOutput()->GetPoints();
    
    //    vtkFloatArray *leftHemiLabels = dynamic_cast<vtkFloatArray*>(lhSphereReader->GetOutput()->GetPointData()->GetScalars());
    //    vtkFloatArray *rightHemiLabels = dynamic_cast<vtkFloatArray*>(rhSphereReader->GetOutput()->GetPointData()->GetScalars());
    // static casting of pointers, hmmm
    vtkIntArray *leftHemiLabels = dynamic_cast<vtkIntArray*>(lhSphereReader->GetOutput()->GetPointData()->GetScalars());
    vtkIntArray *rightHemiLabels = dynamic_cast<vtkIntArray*>(rhSphereReader->GetOutput()->GetPointData()->GetScalars());
    
    vtkCellArray *leftCellArray = lhSphereReader->GetOutput()->GetPolys();
    vtkCellArray *rightCellArray = rhSphereReader->GetOutput()->GetPolys();
    
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
    vtkIdType numberOfPoints = 0;
        
    for (unsigned long int i = 0; i < numberOfPointsInLeftHemi; i++)
      {
        leftHemiPoints->GetPoint(i, point);
        
        for (int j = 0; j < 3; j++)
          {
            point[j] -= 100;
          }
        outputPoints->InsertPoint(i, point);
        outputLabels->SetValue(i, (int)leftHemiLabels->GetTuple1(i));
      }
    
    for (unsigned long int i = 0; i < numberOfPointsInRightHemi; i++)
      {
        rightHemiPoints->GetPoint(i, point);
        
        for (int j = 0; j < 3; j++)
          {
            point[j] += 100;
          }
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

    std::cout << "Output points size=" << outputPoints->GetNumberOfPoints() << std::endl;
    std::cout << "Output triangles size=" << outputTriangles->GetNumberOfCells() << std::endl;

    vtkPolyData *connectivityPolyData = connectivityReader->GetOutput();
    connectivityPolyData->BuildCells();
    connectivityPolyData->BuildLinks();

    vtkPoints *leftWhiteMatterPoints = lhWhiteReader->GetOutput()->GetPoints();
    vtkPoints *rightWhiteMatterPoints = rhWhiteReader->GetOutput()->GetPoints();
    vtkPoints *connectivityPoints = connectivityReader->GetOutput()->GetPoints();

    vtkIdType maxNumberLines = args.maxLines;
    if (maxNumberLines == -1)
      {
        maxNumberLines = connectivityReader->GetOutput()->GetLines()->GetNumberOfCells();
      }
    
    std::cout << "maxNumberLines=" << maxNumberLines << std::endl;
    
    if (args.usePointBased)
      {
        createConnections(true,
                          numberOfPointsInLeftHemi, 
                          numberOfPointsInRightHemi, 
                          leftWhiteMatterPoints,
                          rightWhiteMatterPoints,
                          connectivityPoints,
                          connectivityPolyData,
                          args.distanceThreshold,
                          maxNumberLines*2,
                          outputLines);

        createConnections(false,
                          numberOfPointsInRightHemi, 
                          numberOfPointsInLeftHemi, 
                          rightWhiteMatterPoints,
                          leftWhiteMatterPoints,
                          connectivityPoints,
                          connectivityPolyData,
                          args.distanceThreshold,
                          maxNumberLines*2,
                          outputLines);        
      }
    else
      {
        createConnections2(
                          rightWhiteMatterPoints,
                          leftWhiteMatterPoints,
                          connectivityPoints,
                          connectivityPolyData,
                          args.distanceThreshold,
                          maxNumberLines,
                          outputLines);        
      }

    std::cout << "#input lines = " << connectivityPolyData->GetLines()->GetNumberOfCells() << ", #output lines=" << outputLines->GetNumberOfCells() << ", maxNumberLines=" << maxNumberLines << std::endl;
    
    vtkPolyData *outputPolyData = vtkPolyData::New();
    outputPolyData->SetPoints(outputPoints);
    outputPolyData->GetPointData()->SetScalars(outputLabels);
    if (args.outputTriangles)
      {
        outputPolyData->SetPolys(outputTriangles);    
      }    
    outputPolyData->SetLines(outputLines);

    vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
    writer->SetFileName(args.outputPolyDataFile.c_str());
    writer->SetInput(outputPolyData);
    writer->Update();
    
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
  return EXIT_SUCCESS;
}
