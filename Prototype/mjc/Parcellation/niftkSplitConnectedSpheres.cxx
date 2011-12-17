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
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkPointData.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Used to split out the pairs of connectivity spheres." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -lhs leftHandSurface.vtk -rhs rightHandSurface.vtk -ol outputLeftHandSurface -or outputRightHandSurface [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input VTK Poly Data." << std::endl;
    std::cout << "    -lhs  <filename>        Input left hand surface file, to match whats in i." << std::endl;
    std::cout << "    -rhs  <filename>        Input right hand surface file, to match whats in i." << std::endl;
    std::cout << "    -ol   <filename>        Output left hand surface." << std::endl;
    std::cout << "    -or   <filename>        Output right hand surface." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string inputLeftSurfaceFile;
  std::string inputRightSurfaceFile;
  std::string outputLeftSurfaceFile;
  std::string outputRightSurfaceFile;
};

/**
 * \brief Transform's VTK poly data file by any number of affine transformations.
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
    else if(strcmp(argv[i], "-lhs") == 0){
      args.inputLeftSurfaceFile=argv[++i];
      std::cout << "Set -lhs=" << args.inputLeftSurfaceFile;
    }
    else if(strcmp(argv[i], "-rhs") == 0){
      args.inputRightSurfaceFile=argv[++i];
      std::cout << "Set -rhs=" << args.inputRightSurfaceFile;
    }
    else if(strcmp(argv[i], "-ol") == 0){
      args.outputLeftSurfaceFile=argv[++i];
      std::cout << "Set -ol=" << args.outputLeftSurfaceFile;
    }
    else if(strcmp(argv[i], "-or") == 0){
      args.outputRightSurfaceFile=argv[++i];
      std::cout << "Set -or=" << args.outputRightSurfaceFile;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputLeftSurfaceFile.length() == 0 || args.inputRightSurfaceFile.length() == 0 || args.inputPolyDataFile.length() == 0 ||
      args.outputLeftSurfaceFile.length() == 0 || args.outputRightSurfaceFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *surfaceReader = vtkPolyDataReader::New();
  surfaceReader->SetFileName(args.inputPolyDataFile.c_str());
  surfaceReader->Update();
  
  vtkPolyDataReader *inputLeftSurfaceReader = vtkPolyDataReader::New();
  inputLeftSurfaceReader->SetFileName(args.inputLeftSurfaceFile.c_str());
  inputLeftSurfaceReader->Update();

  vtkPolyDataReader *inputRightSurfaceReader = vtkPolyDataReader::New();
  inputRightSurfaceReader->SetFileName(args.inputRightSurfaceFile.c_str());
  inputRightSurfaceReader->Update();
  
  unsigned long int numberOfPointsInLeftHemi = inputLeftSurfaceReader->GetOutput()->GetNumberOfPoints();
  unsigned long int numberOfPointsInRightHemi = inputRightSurfaceReader->GetOutput()->GetNumberOfPoints();

  std::cout << "#point in left=" << numberOfPointsInLeftHemi << ", #point in right=" << numberOfPointsInRightHemi << std::endl;
  
  vtkIntArray *outputLeftLabels = vtkIntArray::New();
  outputLeftLabels->SetNumberOfComponents(1);
  outputLeftLabels->SetNumberOfValues(numberOfPointsInLeftHemi);

  vtkIntArray *outputRightLabels = vtkIntArray::New();
  outputRightLabels->SetNumberOfComponents(1);
  outputRightLabels->SetNumberOfValues(numberOfPointsInRightHemi);

  vtkIntArray *surfaceLabels = dynamic_cast<vtkIntArray*>(surfaceReader->GetOutput()->GetPointData()->GetScalars());
  
  for (unsigned long int i = 0; i < numberOfPointsInLeftHemi; i++)
    {
      outputLeftLabels->SetValue(i, (int)surfaceLabels->GetTuple1(i));
    }

  for (unsigned long int i = 0; i < numberOfPointsInRightHemi; i++)
    {
      outputRightLabels->SetValue(i, (int)surfaceLabels->GetTuple1(i + numberOfPointsInLeftHemi));
    }

  inputLeftSurfaceReader->GetOutput()->GetPointData()->SetScalars(outputLeftLabels);
  inputRightSurfaceReader->GetOutput()->GetPointData()->SetScalars(outputRightLabels);
  
  vtkPolyDataWriter *leftWriter = vtkPolyDataWriter::New();
  leftWriter->SetFileName(args.outputLeftSurfaceFile.c_str());
  leftWriter->SetInput(inputLeftSurfaceReader->GetOutput());
  leftWriter->Update();

  vtkPolyDataWriter *rightWriter = vtkPolyDataWriter::New();
  rightWriter->SetFileName(args.outputRightSurfaceFile.c_str());
  rightWriter->SetInput(inputRightSurfaceReader->GetOutput());
  rightWriter->Update();

}
