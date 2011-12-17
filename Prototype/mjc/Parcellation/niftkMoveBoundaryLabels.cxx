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
#include "vtkDataArray.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkType.h"
#include "vtkCellTypes.h"
#include "vtkCellLinks.h"
#include "vtkCell.h"
#include "vtkFloatArray.h"
#include <set>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a label connectivity file, and moves the boundary by a number of iterations." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i connectivity.vtk -o output.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i  <filename>          Input VTK Poly Data file containing connectivity lines in mm space, in space of T1 image." << std::endl;
    std::cout << "    -o    <filename>          Output VTK Poly Data file containing spherical surfaces, connectivity lines and labels." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -iters <int> [1]          The number of iterations." << std::endl << std::endl;    
  }


struct arguments
{
  std::string inputConnectivityFile;
  std::string outputConnectivityFile;
  int iterations;
};

vtkPolyData* extractBoundaryModel(vtkPolyData *polyData)
  {
    vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
    vtkPoints *points = polyData->GetPoints();
    vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
    vtkIdType pointNumber = 0;
    vtkIdType cellNumber = 0;
    short unsigned int numberOfCellsUsingCurrentPoint;
    vtkIdType *listOfCellIds;
    vtkIdType *listOfPointIds;
    int currentLabel;
    int nextLabel;
    std::set<int> nextLabels;
    vtkIdType numberOfPointsInCurrentCell;
    double point[3];
    vtkIdType *outputLine = new vtkIdType[2];
    
    vtkPoints *outputPoints = vtkPoints::New();
    outputPoints->SetDataTypeToFloat();
    outputPoints->Allocate(numberOfPoints);
    
    vtkIntArray *outputLabels = vtkIntArray::New();
    outputLabels->SetNumberOfComponents(1);
    outputLabels->SetNumberOfValues(numberOfPoints);
    
    vtkCellArray *outputLines = vtkCellArray::New();
    
    vtkFloatArray *outputVectors = vtkFloatArray::New();
    outputVectors->SetNumberOfComponents(3);
    outputVectors->SetNumberOfValues(numberOfPoints);
    
    unsigned long int numberOfBoundaryPoints = 0;
    unsigned long int numberOfJunctionPoints = 0;
    
    // First we identify junction points, those with at least 3 labels in neighbours.
    for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
      {
        polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

        currentLabel = (int)scalarData->GetTuple1(pointNumber);
        nextLabels.clear();

        // Get all cells containing this point, if triangle, store any surrounding labels that differ.
        for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
          {
            polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                          
            if (numberOfPointsInCurrentCell == 3)
              {
                for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                  {
                    nextLabel = (int)scalarData->GetTuple1(listOfPointIds[i]);
                    if (nextLabel != currentLabel)
                      {
                        nextLabels.insert(nextLabel);
                      }
                  }                  
              }
          }

        // Using nextLabels we know how many labels are in neighbourhood, so we can label points as 
        // 0 = within a region.
        // 1 = on boundary
        // 2 = at junction
        
        points->GetPoint(pointNumber, point);
        outputPoints->InsertPoint(pointNumber, point);
        outputVectors->InsertTuple3(pointNumber, 0, 0, 0);
        
        if (nextLabels.size() == 0)
          {
            // We are in a region
            outputLabels->SetValue(pointNumber, 0);
          }
        else if (nextLabels.size() == 1)
          {
            // We are on boundary
            outputLabels->SetValue(pointNumber, 1);
            numberOfBoundaryPoints++;
          }
        else
          {
            // We are on junction
            outputLabels->SetValue(pointNumber, 2);
            numberOfJunctionPoints++;
          }
      }
    
    // Also, for each boundary or junction point, connect it to all neighbours that are also boundary or junction points.
    for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
      {
        if (outputLabels->GetValue(pointNumber) > 0)
          {
            polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
            
            for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
              {
                polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                
                if (numberOfPointsInCurrentCell == 3)
                  {
                    for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                      {
                        
                        if (listOfPointIds[i] != pointNumber && outputLabels->GetValue(listOfPointIds[i]) > 0)
                          {
                            outputLine[0] = pointNumber;
                            outputLine[1] = listOfPointIds[i];
                            outputLines->InsertNextCell(2, outputLine);
                            
                          }
                      }                    
                  }
              }                    
          }
      }
    
    std::cout << "Number of boundary points=" << numberOfBoundaryPoints << ", number of junction points=" << numberOfJunctionPoints << std::endl;
    
    vtkPolyData *outputPolyData = vtkPolyData::New();
    outputPolyData->SetPoints(outputPoints);
    outputPolyData->GetPointData()->SetScalars(outputLabels);
    outputPolyData->GetPointData()->SetVectors(outputVectors);
    outputPolyData->SetLines(outputLines);
    outputPolyData->BuildCells();
    outputPolyData->BuildLinks();
    
    outputPoints->Delete();
    outputLabels->Delete();
    outputLines->Delete();
    outputVectors->Delete();
    
    return outputPolyData;
  }


bool hasConnectedLines(vtkPolyData* polyData, vtkIdType pointNumber)
{
  bool result = false;
  short unsigned int numberOfCellsUsingCurrentPoint = 0;
  vtkIdType *listOfCellIds;
  vtkIdType *listOfPointIds;
  vtkIdType cellNumber = 0;
  vtkIdType numberOfPointsInCurrentCell = 0;

    polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

    for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
    {
        polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
        if (numberOfPointsInCurrentCell == 2)
        {
          result = true;
          break;
        }
    }
    return result;
}

/**
 * \brief Performs update to parcellation.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.iterations = 1;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputConnectivityFile=argv[++i];
      std::cout << "Set -i=" << args.inputConnectivityFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputConnectivityFile=argv[++i];
      std::cout << "Set -o=" << args.outputConnectivityFile;
    }
    else if(strcmp(argv[i], "-iters") == 0){
      args.iterations=atoi(argv[++i]);
      std::cout << "Set -iters=" << niftk::ConvertToString(args.iterations);
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputConnectivityFile.length() == 0 || args.outputConnectivityFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  // Load poly data containing surface and connections.
  vtkPolyDataReader *surfaceReader = vtkPolyDataReader::New();
  surfaceReader->SetFileName(args.inputConnectivityFile.c_str());
  surfaceReader->Update();

  // Get hold of the point data, number points, scalars etc.
  vtkPolyData *polyData = surfaceReader->GetOutput();
  vtkDataArray *scalarData = polyData->GetPointData()->GetScalars();
  vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
  vtkIdType numberOfCells = polyData->GetNumberOfCells();
  
  polyData->BuildCells();
  polyData->BuildLinks();
  
  if (scalarData == NULL)
    {
      std::cerr << "Couldn't find scalar data (labels)." << std::endl;
      return EXIT_FAILURE;
    }

  // Debug info
  std::cout << "Loaded file " << args.inputConnectivityFile \
    << " containing " << surfaceReader->GetOutput()->GetNumberOfPolys()  \
    << " triangles, and " << surfaceReader->GetOutput()->GetNumberOfLines() \
    << " lines, and " << numberOfPoints \
    << " points, and " << numberOfCells \
    << " cells." \
    << std::endl;

  vtkPolyData *boundaryModel = NULL;
  short unsigned int numberOfCellsUsingCurrentPoint = 0;
  vtkIdType *listOfCellIds;
  int currentLabel = 0;
  int nextLabel = 0;
  int labelThatWeAreMoving = 0;
  std::set<int> nextLabels;
  std::set<int>::iterator nextLabelsIterator;
  vtkIdType cellNumber = 0;
  vtkIdType numberOfPointsInCurrentCell = 0;
  vtkIdType *listOfPointIds;
  vtkIdType nextPointNumber = 0;

  // Work out total number of label values.
  std::set<int> labelNumbers;
  for (vtkIdType pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      labelNumbers.insert((int)scalarData->GetTuple1(pointNumber));
    }
  std::set<int>::iterator labelNumbersIterator;
  std::cout << "Found labels:";
  
  for (labelNumbersIterator = labelNumbers.begin(); labelNumbersIterator != labelNumbers.end(); labelNumbersIterator++)
    {
      std::cout << (*labelNumbersIterator) << " ";
    }
  std::cout << std::endl;
  
  for (int iterationNumber = 0; iterationNumber < args.iterations; iterationNumber++)
    {
      // Extract surface model
      boundaryModel = extractBoundaryModel(polyData);
      
      for(labelNumbersIterator = labelNumbers.begin(); labelNumbersIterator != labelNumbers.end(); labelNumbersIterator++)
        {
          scalarData = dynamic_cast<vtkIntArray*>(polyData->GetPointData()->GetScalars());
          
          // Copy scalar data
          vtkIntArray *newScalarData = vtkIntArray::New();
          newScalarData->SetNumberOfComponents(1);
          newScalarData->SetNumberOfValues(numberOfPoints);
          for (vtkIdType pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
            {
              newScalarData->InsertTuple1(pointNumber, (int)(scalarData->GetTuple1(pointNumber)));
            }
                    
          labelThatWeAreMoving = (*labelNumbersIterator++);
          
          std::cout << "Moving label:" << labelThatWeAreMoving << std::endl;
          
          // Check each point, and only move boundary ones, not junctions.
          for (vtkIdType pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
            {
              currentLabel = (int)scalarData->GetTuple1(pointNumber);

              if (currentLabel == labelThatWeAreMoving && boundaryModel->GetPointData()->GetScalars()->GetTuple1(pointNumber) == 1)
                {
                  polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);
                  
                  nextLabels.clear();

                  // Get neighbouring labels
                  for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
                    {
                      polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
                      
                      if (numberOfPointsInCurrentCell == 3)
                        {
                          for (int i = 0; i < numberOfPointsInCurrentCell; i++)
                            {
                              nextPointNumber = listOfPointIds[i];
                              nextLabel = (int)scalarData->GetTuple1(nextPointNumber);
                              if (nextLabel != currentLabel)
                                {
                                  if (true || hasConnectedLines(polyData, nextPointNumber))
                                    {
                                      nextLabels.insert(nextLabel);    
                                    }
                                }
                            }                  
                          
                        }
                    } // end for each cell, working out neighboring labels.
                  
                  // We should have 1 label
                  if (nextLabels.size() == 1)
                    {
                      nextLabelsIterator = nextLabels.begin();
                      nextLabel = (*nextLabelsIterator);
                      
                      newScalarData->SetTuple1(pointNumber, nextLabel);
                    }
                  
                } // end if boundary point
            } // end for each point
          
          polyData->GetPointData()->SetScalars(newScalarData);
          newScalarData->Delete();
          
        } // end for each label
      
      boundaryModel->Delete();
      
    } // end for each iteration
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputConnectivityFile.c_str());
  writer->SetInput(polyData);
  writer->Update();
  
  return EXIT_SUCCESS;
}
