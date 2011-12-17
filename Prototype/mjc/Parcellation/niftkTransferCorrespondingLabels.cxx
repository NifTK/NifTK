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
    std::cout << "  Transfers label values from one PolyData to another, by taking the corresponding point" << std::endl;
    std::cout << "  This means that both Poly Data files must have the same number of points" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -ref inputPolyData.vtk -data inputPolyData2.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -ref    <filename>      Input VTK Poly Data, to determine geometry" << std::endl;
    std::cout << "    -data   <filename>      Input VTK Poly Data, containing scalar values that will be transfered" << std::endl;
    std::cout << "    -o    <filename>        Output VTK Poly Data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputReferencePolyDataFile;
  std::string inputDataPolyDataFile;
  std::string outputPolyDataFile;
};

/**
 * \brief Drop lines from a poly data.
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
    else if(strcmp(argv[i], "-ref") == 0){
      args.inputReferencePolyDataFile=argv[++i];
      std::cout << "Set -ref=" << args.inputReferencePolyDataFile;
    }
    else if(strcmp(argv[i], "-data") == 0){
      args.inputDataPolyDataFile=argv[++i];
      std::cout << "Set -data=" << args.inputDataPolyDataFile;
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
  if (args.outputPolyDataFile.length() == 0 || args.inputDataPolyDataFile.length() == 0 || args.inputReferencePolyDataFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *referenceReader = vtkPolyDataReader::New();
  referenceReader->SetFileName(args.inputReferencePolyDataFile.c_str());
  referenceReader->Update();

  std::cout << "Loaded PolyData:" << args.inputReferencePolyDataFile << std::endl;
  
  vtkPolyDataReader *dataReader = vtkPolyDataReader::New();
  dataReader->SetFileName(args.inputDataPolyDataFile.c_str());
  dataReader->Update();

  std::cout << "Loaded PolyData:" << args.inputDataPolyDataFile << std::endl;
  
  vtkPoints *referencePoints = referenceReader->GetOutput()->GetPoints();
  vtkPoints *dataPoints = dataReader->GetOutput()->GetPoints();

  if (referencePoints->GetNumberOfPoints() != dataPoints->GetNumberOfPoints())
    {
      std::cerr << "The reference data set has " << referencePoints->GetNumberOfPoints() \
        << " points, whereas the other dataset has " << dataPoints->GetNumberOfPoints() \
        << std::endl;
      return EXIT_FAILURE;
    }

  vtkIntArray *outputLabels = vtkIntArray::New();
  outputLabels->SetNumberOfComponents(1);
  outputLabels->SetNumberOfValues(referencePoints->GetNumberOfPoints());

  vtkIntArray *dataLabels = dynamic_cast<vtkIntArray*>(dataReader->GetOutput()->GetPointData()->GetScalars());

  vtkIdType pointNumber = 0;
  vtkIdType numberOfPoints = referencePoints->GetNumberOfPoints();

  for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      outputLabels->InsertTuple1(pointNumber, dataLabels->GetTuple1(pointNumber));
    }
  
  referenceReader->GetOutput()->GetPointData()->SetScalars(outputLabels);
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->SetInput(referenceReader->GetOutput());
  writer->Update();
  
  return EXIT_SUCCESS;

}
