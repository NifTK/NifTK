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
#include "vtkIntArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataWriter.h"
#include <set>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes two surfaces, with the same number of points and computes SSD on label values." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputSurface1.vtk -j inputSurface2.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i  <filename>          Input VTK Poly Data file." << std::endl;
    std::cout << "    -j  <filename>          Input VTK Poly Data file." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputSurface1;
  std::string inputSurface2;
};

/**
 * \brief Computes SSD on surfaces
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
      args.inputSurface1=argv[++i];
      std::cout << "Set -i=" << args.inputSurface1;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputSurface2=argv[++i];
      std::cout << "Set -j=" << args.inputSurface2;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputSurface1.length() == 0 || args.inputSurface1.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *inputReader1 = vtkPolyDataReader::New();
  inputReader1->SetFileName(args.inputSurface1.c_str());
  inputReader1->Update();
  
  std::cout << "Read:" << args.inputSurface1 << std::endl;
  
  vtkPolyDataReader *inputReader2 = vtkPolyDataReader::New();
  inputReader2->SetFileName(args.inputSurface2.c_str());
  inputReader2->Update();

  std::cout << "Read:" << args.inputSurface2 << std::endl;
  
  vtkPoints *surfacePoints1 = inputReader1->GetOutput()->GetPoints();
  vtkPoints *surfacePoints2 = inputReader2->GetOutput()->GetPoints();
  
  vtkIdType numberOfPointsOnSurface1 = surfacePoints1->GetNumberOfPoints();
  vtkIdType numberOfPointsOnSurface2 = surfacePoints2->GetNumberOfPoints();
    
  std::cout << "#points 1:" << numberOfPointsOnSurface1 << std::endl;
  std::cout << "#points 2:" << numberOfPointsOnSurface2 << std::endl;
  
  if (numberOfPointsOnSurface1 != numberOfPointsOnSurface2)
    {
      std::cerr << "ERROR: Number of points on surface 1 = " << numberOfPointsOnSurface1 << ", points on surface 2 = " << numberOfPointsOnSurface2 << std::endl;
      Usage(argv[0]);
      return EXIT_FAILURE;      
    }

  vtkIntArray *surfaceLabels1 = dynamic_cast<vtkIntArray*>(inputReader1->GetOutput()->GetPointData()->GetScalars());
  vtkIntArray *surfaceLabels2 = dynamic_cast<vtkIntArray*>(inputReader2->GetOutput()->GetPointData()->GetScalars());

  double ssd = 0;
  double diff = 0;
  
  for(vtkIdType i = 0; i < numberOfPointsOnSurface1; i++)
    {
      diff = (int)surfaceLabels1->GetValue(i) - (int)surfaceLabels2->GetValue(i);
      ssd += (diff*diff);
    }

  std::cout << "ssd=" << ssd << std::endl;
  
  return EXIT_SUCCESS;
}
