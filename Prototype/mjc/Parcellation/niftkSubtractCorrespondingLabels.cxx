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
#include "vtkIntArray.h"
#include "vtkPointData.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes two surfaces, of the same number of points, and subtracts the label values" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -j inputPolyData2.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>      Input VTK Poly Data" << std::endl;
    std::cout << "    -j    <filename>      Input VTK Poly Data" << std::endl;
    std::cout << "    -o    <filename>      Output VTK Poly Data where the scalar values are i - j" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyData1;
  std::string inputPolyData2;
  std::string outputPolyData;
};
/**
 * \brief Subtracts scalar values, so, a bit like a difference image.
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
      args.inputPolyData1=argv[++i];
      std::cout << "Set -i=" << args.inputPolyData1;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputPolyData2=argv[++i];
      std::cout << "Set -j=" << args.inputPolyData2;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyData=argv[++i];
      std::cout << "Set -o=" << args.outputPolyData;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputPolyData1.length() == 0 || args.inputPolyData2.length() == 0 || args.outputPolyData.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *reader1 = vtkPolyDataReader::New();
  reader1->SetFileName(args.inputPolyData1.c_str());
  reader1->Update();

  std::cout << "Loaded PolyData:" << args.inputPolyData1 << std::endl;
  
  vtkPolyDataReader *reader2 = vtkPolyDataReader::New();
  reader2->SetFileName(args.inputPolyData2.c_str());
  reader2->Update();

  std::cout << "Loaded PolyData:" << args.inputPolyData2 << std::endl;
  
  vtkPoints *inputPoints1 = reader1->GetOutput()->GetPoints();
  vtkPoints *inputPoints2 = reader2->GetOutput()->GetPoints();
  
  if (inputPoints1->GetNumberOfPoints() != inputPoints2->GetNumberOfPoints())
     {
       std::cerr << "The -i data set has " << inputPoints1->GetNumberOfPoints() \
         << " points, whereas the -j dataset has " << inputPoints2->GetNumberOfPoints() \
         << std::endl;
       return EXIT_FAILURE;
     }

   vtkIntArray *outputLabels = vtkIntArray::New();
   outputLabels->SetNumberOfComponents(1);
   outputLabels->SetNumberOfValues(inputPoints1->GetNumberOfPoints());

   vtkIntArray *input1Labels = dynamic_cast<vtkIntArray*>(reader1->GetOutput()->GetPointData()->GetScalars());
   vtkIntArray *input2Labels = dynamic_cast<vtkIntArray*>(reader2->GetOutput()->GetPointData()->GetScalars());
   
   vtkIdType pointNumber = 0;
   vtkIdType numberOfPoints = inputPoints1->GetNumberOfPoints();

   for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
     {
       outputLabels->InsertTuple1(pointNumber, input1Labels->GetTuple1(pointNumber) - input2Labels->GetTuple1(pointNumber));
     }
   
   reader1->GetOutput()->GetPointData()->SetScalars(outputLabels);
   
   vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
   writer->SetFileName(args.outputPolyData.c_str());
   writer->SetInput(reader1->GetOutput());
   writer->Update();
   
   return EXIT_SUCCESS;

 }
