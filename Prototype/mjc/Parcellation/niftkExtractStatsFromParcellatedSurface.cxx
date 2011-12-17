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
#include <limits>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a parcellated surface (e.g. sphere), and a matching surface containing the same number of points" << std::endl;
    std::cout << "  in the same order, and scalar data such as thickness values, and computes stats for each parcellation label." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -parc inputParcellatedSurface.vtk -data inputDataSurface.vtk [-mean|-sd|-count|-min|-max] [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -parc  <filename>          Input VTK Poly Data file containing scalar labels for each parcellated region." << std::endl;
    std::cout << "    -data  <filename>          Input VTK Poly Data file containing scalar data values such as thickness" << std::endl << std::endl;
    std::cout << "  and one of" << std::endl;
    std::cout << "    -count                     Output number of points in region" << std::endl;
    std::cout << "    -min                       Output minimum of region" << std::endl;
    std::cout << "    -max                       Output maximum of region" << std::endl;    
    std::cout << "    -mean                      Calculate mean of region" << std::endl;
    std::cout << "    -sd                        Calculate standard deviation of region" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -name                      Name of the scalars to assess. Default 'scalars'" << std::endl;
  }

struct arguments
{
  std::string inputParcellatedSurface;
  std::string inputDataSurface;
  bool outputMean;
  bool outputStdDev;
  bool outputCount;
  bool outputMin;
  bool outputMax;  
  std::string scalarName;
};

double calculateMin(std::vector<double>& list)
  {
    double min = std::numeric_limits<double>::max();
    
    std::vector<double>::iterator iterator;
    for (iterator = list.begin(); iterator != list.end(); ++iterator)
      {
        if (*iterator < min)
          {
            min = *iterator;
          }
      }
    return min;
  }

double calculateMax(std::vector<double>& list)
  {
    double max = std::numeric_limits<double>::min();
    
    std::vector<double>::iterator iterator;
    for (iterator = list.begin(); iterator != list.end(); ++iterator)
      {
        if (*iterator > max)
          {
            max = *iterator;
          }
      }
    return max;
  }

double calculateMean(std::vector<double>& list)
  {
    unsigned long int size = list.size();
    std::vector<double>::iterator iterator;
    
    double mean = 0;
    for (iterator = list.begin(); iterator != list.end(); ++iterator)
      {
        mean += (*iterator);
      }
    mean /= (double)size;
    return mean;
  }

double calculateStdDev(std::vector<double>& list)
  {
    unsigned long int size = list.size();
    std::vector<double>::iterator iterator;
    
    double mean = calculateMean(list);
    double stdDev = 0;
    for (iterator = list.begin(); iterator != list.end(); ++iterator)
      {
        stdDev += ((*iterator - mean) * (*iterator - mean));
      }
    stdDev /= (double)(size-1);
    stdDev = sqrt(stdDev);
    return stdDev;
  }

void outputResults(int mode, std::set<int>& setOfLabels, std::map<int, int>& map, std::vector<double>* data)
  {
    int label;
    int index;

    std::set<int>::iterator setIterator;
    std::map<int, int>::iterator mapIterator;
    
    for(setIterator = setOfLabels.begin(); setIterator != setOfLabels.end(); ++setIterator)
      {
        label = *setIterator;
        mapIterator = map.find(label);
        index = (*mapIterator).second;

        if (mode == 0)
          {
            std::cout << data[index].size() << ",";
          }
        else if (mode == 1)
          {
            std::cout << calculateMin(data[index]) << "," ;
          }
        else if (mode == 2)
          {
            std::cout << calculateMax(data[index]) << "," ;
          }
        else if (mode == 3)
          {
            std::cout << calculateMean(data[index]) << "," ;
          }
        else if (mode == 4)
          {
            std::cout << calculateStdDev(data[index]) << "," ;
          }
      }  
    std::cout << std::endl;
  }
/**
 * \brief Combines various VTK poly data file into one connectivity file.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.outputMean = false;
  args.outputStdDev = false;
  args.outputCount = false;
  args.outputMin = false;
  args.outputMax = false;
  args.scalarName = "scalars";
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-parc") == 0){
      args.inputParcellatedSurface=argv[++i];
      std::cout << "Set -parc=" << args.inputParcellatedSurface;
    }
    else if(strcmp(argv[i], "-data") == 0){
      args.inputDataSurface=argv[++i];
      std::cout << "Set -data=" << args.inputDataSurface;
    }
    else if(strcmp(argv[i], "-mean") == 0){
      args.outputMean=true;
      std::cout << "Set -mean=" << niftk::ConvertToString(args.outputMean);
    }        
    else if(strcmp(argv[i], "-sd") == 0){
      args.outputStdDev=true;
      std::cout << "Set -sd=" << niftk::ConvertToString(args.outputStdDev);
    }        
    else if(strcmp(argv[i], "-count") == 0){
      args.outputCount=true;
      std::cout << "Set -count=" << niftk::ConvertToString(args.outputCount);
    }        
    else if(strcmp(argv[i], "-min") == 0){
      args.outputMin=true;
      std::cout << "Set -min=" << niftk::ConvertToString(args.outputMin);
    }        
    else if(strcmp(argv[i], "-max") == 0){
      args.outputMax=true;
      std::cout << "Set -max=" << niftk::ConvertToString(args.outputMax);
    }   
    else if(strcmp(argv[i], "-name") == 0){
      args.scalarName=argv[++i];
      std::cout << "Set -name=" << args.scalarName;
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.inputDataSurface.length() == 0 || args.inputParcellatedSurface.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (args.outputCount == false  
      && args.outputMax == false 
      && args.outputMean == false 
      && args.outputMin == false 
      && args.outputStdDev == false )
    {
      std::cerr << argv[0] << ":\tYou must specify at least one of the output options -count, -min, -max, -mean or -sd " << std::endl;
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *parcalletedSurfaceReader = vtkPolyDataReader::New();
  parcalletedSurfaceReader->SetFileName(args.inputParcellatedSurface.c_str());
  parcalletedSurfaceReader->Update();
  
  std::cout << "Read:" << args.inputParcellatedSurface << std::endl;
  
  vtkPolyDataReader *dataSurfaceReader = vtkPolyDataReader::New();
  dataSurfaceReader->SetFileName(args.inputDataSurface.c_str());
  dataSurfaceReader->Update();

  std::cout << "Read:" << args.inputDataSurface << std::endl;
  
  vtkPoints *parcellationPoints = parcalletedSurfaceReader->GetOutput()->GetPoints();
  vtkPoints *dataPoints = dataSurfaceReader->GetOutput()->GetPoints();
  
  vtkIdType numberOfPointsOnParcellatedSurface = parcellationPoints->GetNumberOfPoints();
  vtkIdType numberOfPointsOnDataSurface = dataPoints->GetNumberOfPoints();
  
  std::cout << "#points parcellated:" << numberOfPointsOnParcellatedSurface << std::endl;
  std::cout << "#points data:" << numberOfPointsOnDataSurface << std::endl;
  
  if (numberOfPointsOnParcellatedSurface != numberOfPointsOnDataSurface)
    {
      std::cerr << "ERROR: Number of points on parcellated surface = " << numberOfPointsOnParcellatedSurface << ", whereas, number of points on data surface = " << numberOfPointsOnDataSurface << std::endl;
      Usage(argv[0]);
      return EXIT_FAILURE;      
    }
  
  // Get a set of all possible labels.
  vtkIntArray *labelValues = dynamic_cast<vtkIntArray*>(parcalletedSurfaceReader->GetOutput()->GetPointData()->GetScalars());
  std::set<int> setOfLabels;
  for (vtkIdType i = 0; i < numberOfPointsOnParcellatedSurface; i++)
    {
      setOfLabels.insert((int)labelValues->GetTuple1(i));
    }
  
  std::cout << "Read " << setOfLabels.size() << " labels" << std::endl;
  
  // Print em out. We also create a map of label value to index, as label values may be non-continous.
  // The set enforces the ordering.
  
  std::map<int, int> map;
  std::set<int>::iterator setIterator;
  std::cout << "Labels are:";
  unsigned long int i = 0;
  for(setIterator = setOfLabels.begin(); setIterator != setOfLabels.end(); ++setIterator)
    {
      std::cout << *setIterator << ",";
      map.insert(std::pair<int, int>(*setIterator, i));
      i++;
    }
  std::cout << std::endl;
  
  // Create array of vectors of doubles.
  std::vector<double> *array = new std::vector<double>[setOfLabels.size()];

  // Only select the right dataset.
  dataSurfaceReader->GetOutput()->GetPointData()->SetActiveScalars(args.scalarName.c_str());

  // Cheesy hack to get round the fact that we might have float/int data.
  // TODO: Do this properly.
  bool isFloat = false;
  vtkFloatArray *floatDataValues = dynamic_cast<vtkFloatArray*>(dataSurfaceReader->GetOutput()->GetPointData()->GetScalars());
  vtkIntArray *intDataValues = dynamic_cast<vtkIntArray*>(dataSurfaceReader->GetOutput()->GetPointData()->GetScalars());
  
  if (floatDataValues != NULL)
    {
      isFloat = true;
    }
  
  int label;
  double scalar;
  int index;
  
  std::map<int, int>::iterator mapIterator;
  
  for(vtkIdType i = 0; i < numberOfPointsOnParcellatedSurface; i++)
    {
      label = (int)labelValues->GetTuple1(i);
      if (isFloat)
        {
          scalar = (double)floatDataValues->GetTuple1(i);    
        }
      else
        {
          scalar = (double)intDataValues->GetTuple1(i);    
        }
      
      mapIterator = map.find(label);
      index = (*mapIterator).second;
      array[index].push_back(scalar);
    }

  // Now print out results
  
  if (args.outputCount)
    {
      outputResults(0, setOfLabels, map, array);
    }
  
  if (args.outputMin)
    {
      outputResults(1, setOfLabels, map, array);
    }
  
  if (args.outputMax)
    {
      outputResults(2, setOfLabels, map, array);
    }
  
  if (args.outputMean)
    {
      outputResults(3, setOfLabels, map, array);
    }
  
  if (args.outputStdDev)
    {
      outputResults(4, setOfLabels, map, array);
    }
  
  return EXIT_SUCCESS;
}
