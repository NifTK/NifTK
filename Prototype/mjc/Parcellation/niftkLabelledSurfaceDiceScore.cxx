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
#include "vtkPointData.h"
#include <set>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes two surfaces and assumes that the scalar values are segmentation labels, so computes dice scores." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData1.vtk -j inputPolyData2.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input VTK Poly Data 1." << std::endl;
    std::cout << "    -j    <filename>        Input VTK Poly Data 2." << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile1;
  std::string inputPolyDataFile2;
};


/**
 * \brief Computes Dice scores from labelled surfaces
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
      args.inputPolyDataFile1=argv[++i];
      std::cout << "Set -i=" << args.inputPolyDataFile1;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputPolyDataFile2=argv[++i];
      std::cout << "Set -j=" << args.inputPolyDataFile2;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputPolyDataFile1.length() == 0 || args.inputPolyDataFile1.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  // Read both surfaces
  vtkPolyDataReader *surface1Reader = vtkPolyDataReader::New();
  surface1Reader->SetFileName(args.inputPolyDataFile1.c_str());
  surface1Reader->Update();
  
  std::cout << "Read:" << args.inputPolyDataFile1 << std::endl;
  
  vtkPolyDataReader *surface2Reader = vtkPolyDataReader::New();
  surface2Reader->SetFileName(args.inputPolyDataFile2.c_str());
  surface2Reader->Update();
  
  std::cout << "Read:" << args.inputPolyDataFile2 << std::endl;

  // Check same number of points
  vtkPoints *surface1Points = surface1Reader->GetOutput()->GetPoints();
  vtkPoints *surface2Points = surface2Reader->GetOutput()->GetPoints();
  
  vtkIdType numberPointsSurface1 = surface1Points->GetNumberOfPoints();
  vtkIdType numberPointsSurface2 = surface2Points->GetNumberOfPoints();
  
  if (numberPointsSurface1 != numberPointsSurface2)
    {
      std::cerr << "ERROR:Surface 1 has " << numberPointsSurface1 << ", but surface 2 has " << numberPointsSurface2 << std::endl;
    }
  
  // Extract the labels

  // Get a set of all possible labels.
  vtkIntArray *surface1Labels = dynamic_cast<vtkIntArray*>(surface1Reader->GetOutput()->GetPointData()->GetScalars());
  vtkIntArray *surface2Labels = dynamic_cast<vtkIntArray*>(surface2Reader->GetOutput()->GetPointData()->GetScalars());
  
  std::set<int> setOfLabels;
  for (vtkIdType i = 0; i < numberPointsSurface1; i++)
    {
      setOfLabels.insert((int)surface1Labels->GetTuple1(i));
    }
  
  std::cout << "Read " << setOfLabels.size() << " labels" << std::endl;

  double meanDiceScore = 0;
  
  std::set<int>::iterator setIterator;
  for (setIterator = setOfLabels.begin(); setIterator != setOfLabels.end(); setIterator++)
    {
      int labelToTest = (*setIterator);
      unsigned long int truePositive = 0;
      unsigned long int trueNegative = 0;
      unsigned long int falsePositive = 0;
      unsigned long int falseNegative = 0;
      int currentLabel1 = 0;
      int currentLabel2 = 0;
      
      for (vtkIdType i = 0; i < numberPointsSurface1; i++)
        {
          currentLabel1 = (int)surface1Labels->GetTuple1(i);
          currentLabel2 = (int)surface2Labels->GetTuple1(i);
          
          if (currentLabel1 == labelToTest)
            {
              if (currentLabel2 == labelToTest)
                {
                  truePositive++;
                }
              else
                {
                  falseNegative++;
                }
            }
          else
            {
              if (currentLabel2 == labelToTest)
                {
                  falsePositive++;
                }
              else
                {
                  trueNegative++;
                }
            }
        } // end for
      
      // Output dice score
      double dice = ((double)truePositive)/(0.5*((double)truePositive + (double)falseNegative + (double)truePositive + (double)falsePositive));
      meanDiceScore += dice;
      
      std::cout << dice << " ";
    }
  meanDiceScore /=(double) setOfLabels.size();
  
  std::cout << meanDiceScore << " ";
  std::cout << std::endl;
}
