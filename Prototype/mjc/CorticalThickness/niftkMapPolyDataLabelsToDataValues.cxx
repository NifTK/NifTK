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
#include "vtkIntArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a VTK Poly Data, with integer label values, and a plain text lookup table, and maps labels to data values." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -j inputLookupTable -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>      Input VTK Poly Data" << std::endl;
    std::cout << "    -j    <filename>      Input plain text lookup table" << std::endl;
    std::cout << "    -o    <filename>      Output VTK Poly Data" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -unknown <float> [0]  Data value to use if label is not in map" << std::endl;
  }

struct arguments
{
  std::string inputPolyData;
  std::string inputTextFile;
  std::string outputPolyData;
  float unknown;
};

/**
 * \brief Subtracts scalar values, so, a bit like a difference image.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.unknown = 0;
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputPolyData=argv[++i];
      std::cout << "Set -i=" << args.inputPolyData;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputTextFile=argv[++i];
      std::cout << "Set -j=" << args.inputTextFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyData=argv[++i];
      std::cout << "Set -o=" << args.outputPolyData;
    }
    else if(strcmp(argv[i], "-unknown") == 0){
      args.unknown=atof(argv[++i]);
      std::cout << "Added -unknown=" << niftk::ConvertToString(args.unknown);
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputPolyData.length() == 0 || args.inputTextFile.length() == 0 || args.outputPolyData.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  vtkPolyDataReader *reader1 = vtkPolyDataReader::New();
  reader1->SetFileName(args.inputPolyData.c_str());
  reader1->Update();

  std::cout << "Loaded " << args.inputPolyData << std::endl;
  
  // Load plain text file int map
  std::map<int, double> map;
  std::map<int, double>::iterator mapIterator;

  double data;
  int label;
  bool nextIsLabel = true;

  std::ifstream inputFile;
  inputFile.open(args.inputTextFile.c_str(), std::ifstream::in); 

  while (!inputFile.eof())
    {
      inputFile >> data;
      if (!inputFile.fail())
      {
        if (nextIsLabel)
          {
            label = (int)data;
            nextIsLabel = false;
          }
        else
          {
            map.insert(std::pair<int, double>(label,data));
            std::cout << "Label " << label << ", maps to " << data << std::endl;
            nextIsLabel = true;
          }
      }
    }
  inputFile.close();

  vtkPoints *dataPoints = reader1->GetOutput()->GetPoints();
  vtkIdType numberOfPointsOnDataSurface = dataPoints->GetNumberOfPoints();

  vtkFloatArray *outputFloatScalars = vtkFloatArray::New();
  outputFloatScalars->SetNumberOfComponents(1);
  outputFloatScalars->SetNumberOfValues(numberOfPointsOnDataSurface);

  vtkDataArray *input1Labels = reader1->GetOutput()->GetPointData()->GetScalars();

  vtkIdType pointNumber = 0;
  for (pointNumber = 0; pointNumber < numberOfPointsOnDataSurface; pointNumber++)
    {
      mapIterator = map.find((int)input1Labels->GetTuple1(pointNumber));
      if (mapIterator != map.end())
        {
          outputFloatScalars->InsertTuple1(pointNumber, (*mapIterator).second); 
        }
      else
        {
          outputFloatScalars->InsertTuple1(pointNumber, args.unknown);
        }
    }
  reader1->GetOutput()->GetPointData()->SetScalars(outputFloatScalars);
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputPolyData.c_str());
  writer->SetInput(reader1->GetOutput());
  writer->Update();
  
}
