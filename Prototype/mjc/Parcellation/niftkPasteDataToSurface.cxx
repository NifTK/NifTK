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
#include "vtkType.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataWriter.h"
#include <limits>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes two poly data, one a reference surface, and one a pointset with scalar data." << std::endl;
    std::cout << "  For each point on the reference surface, will find the closest point on the other surface, and copy the scalar data." << std::endl; 
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -ref inputPolyData.vtk -data dataSurface.vtk -o resultSurface.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -ref  <filename>        Input VTK Poly Data reference surface" << std::endl;
    std::cout << "    -data <filename>        Input VTK Poly Data, containing some data, such as thickness values at each point" << std::endl;
    std::cout << "    -o    <filename>        Output VTK Poly Data" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -sameOrder              If data is exactly the same order, and size, then we can assume pointwise correspondence." << std::endl << std::endl;
    std::cout << "    -outputInt              Output an int array rather than a float array" << std::endl;
  }

struct arguments
{
  std::string inputReferenceFile;
  std::string inputDataFile;
  std::string outputFile;
  bool sameOrder;
  bool outputIsInt;
};

double distanceBetweenPoints(double* a, double *b)
  {
    return (
          ((a[0]-b[0]) * (a[0]-b[0])) 
        + ((a[1]-b[1]) * (a[1]-b[1]))
        + ((a[2]-b[2]) * (a[2]-b[2]))
        );
  }

vtkIdType getClosestPoint(
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


/**
 * \brief Transform's VTK poly data file by any number of affine transformations.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.sameOrder = false;
  args.outputIsInt = false;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-ref") == 0){
      args.inputReferenceFile=argv[++i];
      std::cout << "Set -ref=" << args.inputReferenceFile;
    }
    else if(strcmp(argv[i], "-data") == 0){
      args.inputDataFile=argv[++i];
      std::cout << "Set -data=" << args.inputDataFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputFile=argv[++i];
      std::cout << "Set -o=" << args.outputFile;
    }
    else if(strcmp(argv[i], "-sameOrder") == 0){
      args.sameOrder=true;
      std::cout << "Set -sameOrder=" << niftk::ConvertToString(args.sameOrder);
    }
    else if(strcmp(argv[i], "-outputInt") == 0){
      args.outputIsInt=true;
      std::cout << "Set -outputInt=" << niftk::ConvertToString(args.outputIsInt);
    }        
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputDataFile.length() == 0 || args.inputReferenceFile.length() == 0 || args.outputFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *referenceSurfaceReader = vtkPolyDataReader::New();
  referenceSurfaceReader->SetFileName(args.inputReferenceFile.c_str());
  referenceSurfaceReader->Update();
  
  vtkPolyDataReader *dataSurfaceReader = vtkPolyDataReader::New();
  dataSurfaceReader->SetFileName(args.inputDataFile.c_str());
  dataSurfaceReader->Update();
  
  bool isFloatData = false;
  
  vtkFloatArray *floatDataValues = dynamic_cast<vtkFloatArray*>(dataSurfaceReader->GetOutput()->GetPointData()->GetScalars());
  vtkIntArray *intDataValues = dynamic_cast<vtkIntArray*>(dataSurfaceReader->GetOutput()->GetPointData()->GetScalars());
  
  if (floatDataValues != NULL)
    {
      isFloatData = true;
    }
  
  vtkPoints *referencePoints = referenceSurfaceReader->GetOutput()->GetPoints();
  vtkPoints *dataPoints = dataSurfaceReader->GetOutput()->GetPoints();
  
  vtkIdType numberOfPointsOnReferenceSurface = referencePoints->GetNumberOfPoints();
  vtkIdType numberOfPointsOnDataSurface = dataPoints->GetNumberOfPoints();
  
  std::cout << "Number of points on ref=" << numberOfPointsOnReferenceSurface << ", number of points on data surface=" << numberOfPointsOnDataSurface << std::endl;
  bool sameNumberOfPoints = (numberOfPointsOnReferenceSurface == numberOfPointsOnDataSurface);
  
  vtkFloatArray *outputFloatScalars = vtkFloatArray::New();
  outputFloatScalars->SetNumberOfComponents(1);
  outputFloatScalars->SetNumberOfValues(numberOfPointsOnReferenceSurface);

  vtkIntArray *outputIntScalars = vtkIntArray::New();
  outputIntScalars->SetNumberOfComponents(1);
  outputIntScalars->SetNumberOfValues(numberOfPointsOnReferenceSurface);

  vtkIdType i;
  vtkIdType index;
  double referencePoint[3];
  double tmp;
  int nextTarget = 10;
  
  for (i = 0; i < numberOfPointsOnReferenceSurface; i++)
    {
      if (args.sameOrder && sameNumberOfPoints)
        {
          if (isFloatData)
            {
              if (args.outputIsInt)
                {
                  outputIntScalars->SetTuple1(i, floatDataValues->GetTuple1(i));    
                }
              else
                {
                  outputFloatScalars->SetTuple1(i, floatDataValues->GetTuple1(i));    
                }
            }
          else
            {
              if (args.outputIsInt)
                {
                  outputIntScalars->SetTuple1(i, intDataValues->GetTuple1(i));    
                }
              else
                {
                  outputFloatScalars->SetTuple1(i, intDataValues->GetTuple1(i));    
                }
            }
                 
        }
      else
        {
          referencePoints->GetPoint(i, referencePoint);
          index = getClosestPoint(dataPoints, referencePoint);
          
          if (isFloatData)
            {
              if (args.outputIsInt)
                {
                  outputIntScalars->SetTuple1(i, floatDataValues->GetTuple1(index));    
                }
              else
                {
                  outputFloatScalars->SetTuple1(i, floatDataValues->GetTuple1(index));
                }
            }
          else
            {
              if (args.outputIsInt)
                {
                  outputIntScalars->SetTuple1(i, intDataValues->GetTuple1(index));    
                }
              else
                {
                  outputFloatScalars->SetTuple1(i, intDataValues->GetTuple1(index));    
                }
            }
        }

      tmp = (double)i*100.0/(double)numberOfPointsOnReferenceSurface;
      
      if ((int)tmp == nextTarget)
        {
          std::cout << "Completed:" << (int)tmp << "%" << std::endl;
          nextTarget += 10;
        }
    }
  
  if (args.outputIsInt)
    {
      referenceSurfaceReader->GetOutput()->GetPointData()->SetScalars(outputIntScalars);    
    }
  else
    {
      referenceSurfaceReader->GetOutput()->GetPointData()->SetScalars(outputFloatScalars);
    }
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputFile.c_str());
  writer->SetInput(referenceSurfaceReader->GetOutput());
  writer->Update();
}
