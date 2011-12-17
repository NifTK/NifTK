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
#include "itkCommandLineHelper.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkFloatArray.h"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
#include "vtkType.h"

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes an ASCII Free Surfer thickness file, and converts it to a VTK Poly Data file, with points, and a scalar value. " << std::endl;
    std::cout << "  This program doesn't do any transformation. So the output is still in FreeSurfer RAS voxel coordinates. " << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputfile.asc -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input text file. See mris_convert to convert lh/rh.thickness to ASCII" << std::endl;
    std::cout << "    -o    <filename>        Output VTK poly data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputFile;
  std::string outputFile;
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
      args.inputFile=argv[++i];
      std::cout << "Set -i=" << args.inputFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputFile=argv[++i];
      std::cout << "Set -o=" << args.outputFile;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }      
  }

  // Validate command line args
  if (args.inputFile.length() == 0 || args.outputFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  FILE* ip;
  char lineOfText[256];
  char numberString[256];
  float point[3];
  float thickness;
  unsigned long int numberOfPoints = 0;
  
  std::vector<double> xList, yList, zList, thicknessList;
  
  xList.clear();
  yList.clear();
  zList.clear();
  thicknessList.clear();
  
  if ((ip = fopen(args.inputFile.c_str(), "r")) == NULL)
    {
      std::cerr <<"Failed to open file:" << args.inputFile << " for reading";
      return EXIT_FAILURE;
    }

  while(fgets(lineOfText, 255, ip) != NULL)
    {
      sscanf(lineOfText, "%s %f %f %f %f\n", numberString, &point[0], &point[1], &point[2], &thickness);
      xList.push_back(point[0]);
      yList.push_back(point[1]);
      zList.push_back(point[2]);
      thicknessList.push_back(thickness);
      
      numberOfPoints++;
    }
  
  std::cout << "Read " << numberOfPoints << " points" << std::endl;
  
  vtkPoints *outputPoints = vtkPoints::New();
  outputPoints->SetDataTypeToFloat();
  outputPoints->Allocate(numberOfPoints);

  vtkFloatArray *outputThickness = vtkFloatArray::New();
  outputThickness->SetNumberOfComponents(1);
  outputThickness->SetNumberOfValues(numberOfPoints);

  for (unsigned long int i = 0; i < numberOfPoints; i++)
    {
      outputPoints->InsertPoint(i, xList[i], yList[i] , zList[i]);
      outputThickness->InsertValue(i, thicknessList[i]);
    }

  vtkPolyData *outputPolyData = vtkPolyData::New();
  outputPolyData->SetPoints(outputPoints);
  outputPolyData->GetPointData()->SetScalars(outputThickness);

  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputFile.c_str());
  writer->SetInput(outputPolyData);
  writer->Update();
  
  if (ip != NULL)
    {
      fclose(ip);
    }
  else
    {
      std::cerr << "Wierd, my input file pointer is NULL, this shouldn't happen, but I'm exiting anyway";
    }
  
  
}
