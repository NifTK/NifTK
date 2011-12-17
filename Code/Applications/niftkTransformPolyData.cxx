/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $

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
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "itkEulerAffineTransform.h"
#include "itkMatrix.h"
#include "vtkIndent.h"
#include "vtkTransformPolyDataFilter.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Transform's a VTK Poly Data file by any number of affine transformations." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input VTK Poly Data." << std::endl;
    std::cout << "    -o    <filename>        Output VTK Poly Data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -t    <filename>        Affine transform file. Text file, 4 rows of 4 numbers. This option can be repeated." << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  std::vector<std::string> transformNames;
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
      std::cout << "Set -i=" << args.inputPolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyDataFile=argv[++i];
      std::cout << "Set -o=" << args.outputPolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      args.transformNames.push_back(argv[++i]);
      std::cout << "Added -t=" << argv[i] << std::endl;
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

  typedef itk::EulerAffineTransform<double, 3, 3> AffineTransformType;
  typedef itk::Matrix<double, 4, 4> AffineMatrixType; 
  
  AffineMatrixType itkMatrix;
  AffineTransformType::Pointer itkTransform = AffineTransformType::New();
  
  vtkMatrix4x4 *vtkMatrix = vtkMatrix4x4::New();
  vtkTransform *vtkTransform = vtkTransform::New();
  vtkTransform->Identity();
  vtkTransform->PostMultiply();
  
  // Read transformations.
  unsigned int i, j, k;
  
  for (i = 0; i < args.transformNames.size(); i++)
    {
      itkTransform->LoadFullAffineMatrix(args.transformNames[i]);
      itkMatrix = itkTransform->GetFullAffineMatrix();
      
      for (j = 0; j < 4; j++)
        {
          for (k = 0; k < 4; k++)
            {
              vtkMatrix->SetElement(j,k,itkMatrix(j,k));
            }
        }
      vtkTransform->Concatenate(vtkMatrix);
    }
  
  std::cout << "Loading PolyData:" << args.inputPolyDataFile << std::endl;
  
  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(args.inputPolyDataFile.c_str());
  reader->Update();
  
  std::cout << "Loaded PolyData:" << args.inputPolyDataFile << std::endl;
  
  vtkTransformPolyDataFilter *filter = vtkTransformPolyDataFilter::New();
  filter->SetInput(reader->GetOutput());
  filter->SetTransform(vtkTransform);
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->Update();
}

