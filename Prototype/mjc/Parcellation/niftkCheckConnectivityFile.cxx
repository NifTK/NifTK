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
#include "vtkType.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include <set>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Checks the connectivity file." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input VTK Poly Data file representing the connectivity of two spheres." << std::endl;
    std::cout << "    -o    <filename>        Output VTK Poly Data where each point contains a scalar showing how many points it's connected to" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
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
    
    std::cerr << "Number of boundary points=" << numberOfBoundaryPoints << ", number of junction points=" << numberOfJunctionPoints << std::endl;
    
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

/**
 * \brief Check connectivity.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputPolyDataFile=argv[++i];
      std::cout << "Set -i=" << args.inputPolyDataFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyDataFile=argv[++i];
      std::cout << "Set -o=" << args.outputPolyDataFile;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.outputPolyDataFile.length() == 0 || args.inputPolyDataFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(args.inputPolyDataFile.c_str());
  reader->Update();

  std::cout << "Loaded PolyData:" << args.inputPolyDataFile << std::endl;

  vtkPolyData *polyData = reader->GetOutput();
  polyData->BuildCells();
  polyData->BuildLinks();
  
  vtkIdType numberOfPoints = polyData->GetNumberOfPoints();
  vtkIdType numberOfLines = polyData->GetLines()->GetNumberOfCells();
  vtkIdType numberOfTriangles = polyData->GetPolys()->GetNumberOfCells();
  
  std::cout << "Number points:" << numberOfPoints << std::endl;
  std::cout << "Number lines:" << numberOfLines << std::endl;
  std::cout << "Number triangles:" << numberOfTriangles << std::endl;

  vtkIdType pointNumber;
  vtkIdType cellNumber;
  vtkIdType *listOfCellIds;
  vtkIdType *listOfPointIds;
  vtkIdType numberOfPointsInCurrentCell;
  short unsigned int numberOfCellsUsingCurrentPoint;
  int numberOfSurroundingTriangles;
  int numberOfConnections;
  
  vtkIntArray *numberOfConnectionsArray = vtkIntArray::New();
  numberOfConnectionsArray->SetNumberOfComponents(1);
  numberOfConnectionsArray->SetNumberOfValues(numberOfPoints);
  numberOfConnectionsArray->SetName("Connections");
  
  vtkIntArray *numberOfTrianglesArray = vtkIntArray::New();
  numberOfTrianglesArray->SetNumberOfComponents(1);
  numberOfTrianglesArray->SetNumberOfValues(numberOfPoints);
  numberOfTrianglesArray->SetName("Triangles");
  
  int simpleHistogram[51];
  for (unsigned int i = 0; i < 51; i++)
    {
      simpleHistogram[i] = 0;
    }
  
  for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      polyData->GetPointCells(pointNumber, numberOfCellsUsingCurrentPoint, listOfCellIds);

      numberOfSurroundingTriangles = 0;
      numberOfConnections = 0;

      for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
        {
          polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
          if (numberOfPointsInCurrentCell == 3)
            {
              numberOfSurroundingTriangles++;
            }
        }
      
      for (cellNumber = 0; cellNumber < numberOfCellsUsingCurrentPoint; cellNumber++)
        {
          polyData->GetCellPoints(listOfCellIds[cellNumber], numberOfPointsInCurrentCell, listOfPointIds);
          
          if (numberOfPointsInCurrentCell == 2)
            {
              numberOfConnections++;
            }
        }

      if (numberOfConnections >= 50)
        {
          simpleHistogram[50]++;
        }
      else
        {
          simpleHistogram[numberOfConnections]++;
        }
      
      numberOfConnectionsArray->InsertValue(pointNumber, numberOfConnections);
      numberOfTrianglesArray->InsertValue(pointNumber, numberOfSurroundingTriangles);
      
    }

  vtkPolyData *boundaryModel = extractBoundaryModel(polyData);
  
  int numberOfBoundaryPoints = 0;
  int numberOfBoundaryPointsWithConnections = 0;
  vtkDataArray *boundaryData = boundaryModel->GetPointData()->GetScalars();
  
  for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      if (boundaryData->GetTuple1(pointNumber) > 0)
        {
          numberOfBoundaryPoints++;
          if (numberOfConnectionsArray->GetTuple1(pointNumber) > 0)
            {
              numberOfBoundaryPointsWithConnections++;
            }
        }
    }
  
  std::cout << "Number of connected boundary points:" << numberOfBoundaryPointsWithConnections << " = " << 100.0*numberOfBoundaryPointsWithConnections/(double)numberOfBoundaryPoints << "%" << std::endl;

  polyData->GetPointData()->AddArray(numberOfConnectionsArray);
  polyData->GetPointData()->AddArray(numberOfTrianglesArray);
  polyData->GetPointData()->RemoveArray("scalars");
  polyData->GetLines()->SetNumberOfCells(0);
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->SetInput(polyData);
  writer->Update();
  
  for (unsigned int i = 0; i < 51; i++)
    {
      std::cout << i << "," << simpleHistogram[i] << std::endl;
    }
  
  return EXIT_SUCCESS;
}
