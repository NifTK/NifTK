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
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDSSegmentationFusionImageFilter.h"
  
/**
 * Basic tests for ShapeBasedAveragingImageFilter
 */
int main(int argc, char * argv[])
{
  // Declare the types of the images
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::DSSegmentationFusionImageFilter<ImageType, ImageType> FilterType;
  typedef itk::ImageFileReader<ImageType> ReaderType; 
  typedef itk::ImageFileWriter<ImageType> WriterType; 
  typedef itk::ImageFileWriter<FilterType::ConflictImageType> ConflictWriterType; 

  ImageType::DirectionType direction; 
  direction[0][0] = 1;
  direction[0][1] = 0;
  direction[0][2] = 0;
  direction[1][0] = 0;
  direction[1][1] = 1;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  
  try
  {
    FilterType::Pointer filter = FilterType::New(); 
    
    char *reliabilityArg = new char[strlen(argv[3]) + 1]; 
    strcpy(reliabilityArg, argv[3]);

    std::vector<double> reliability; 
    
    // Parse the reliability. 
    int count = 0; 
    char* pch = strtok(reliabilityArg," ");
    bool isAllZero = true; 
    while (pch != NULL)
    {
      filter->SetSegmentationReliability(count, atof(pch)); 
      if (isAllZero)
      {
        if (atof(pch) != 0.0)
          isAllZero = false; 
      }
      pch = strtok (NULL, " ");
      count++; 
    }
    if (reliabilityArg != NULL) delete reliabilityArg;

    // If they are all zero, they are image names. 
    ReaderType::Pointer *reader2 = new ReaderType::Pointer[count]; 
    count = 0; 
    if (isAllZero)
    {
      char *reliabilityArg2 = new char[strlen(argv[3]) + 1]; 
      strcpy(reliabilityArg2, argv[3]); 
      
      char* pch = strtok(reliabilityArg2," ");
      while (pch != NULL)
      {
        reader2[count] = ReaderType::New(); 
        reader2[count]->SetFileName(pch); 
        reader2[count]->Update(); 
        reader2[count]->GetOutput()->SetDirection(direction); 
        
        if (count == 0)
          filter->SetTargetImage(reader2[count]->GetOutput()); 
        else
          filter->SetRegisteredAtlases(count-1, reader2[count]->GetOutput()); 
        pch = strtok (NULL, " ");
        count++; 
      }
      if (reliabilityArg2 != NULL) delete reliabilityArg2;

    }
  
    int startingIndex = 9; 
    ReaderType::Pointer *reader1 = new ReaderType::Pointer[argc-startingIndex]; 
    
    for (int i = startingIndex; i < argc; i++)
    {
      reader1[i-startingIndex] = ReaderType::New(); 
      reader1[i-startingIndex]->SetFileName(argv[i]); 
      reader1[i-startingIndex]->Update(); 
      reader1[i-startingIndex]->GetOutput()->SetDirection(direction);       
    
      std::cout << "Setting input file: " << argv[i] << std::endl; 
      filter->SetInput(i-startingIndex, reader1[i-startingIndex]->GetOutput()); 
    }
    
    filter->SetNumberOfThreads(1); 
    filter->SetReliabilityMode(atoi(argv[4])); 
    filter->SetLocalRegionRadius(atoi(argv[5])); 
    filter->SetCombinationMode(atoi(argv[6])); 
    filter->SetGain(atof(argv[7])); 
    filter->SetPlausibilityThreshold(atof(argv[8])); 
    filter->Update(); 
    
    WriterType::Pointer writer = WriterType::New(); 
    writer->SetInput(filter->GetOutput()); 
    writer->SetFileName(argv[1]); 
    writer->Update(); 
    
    ConflictWriterType::Pointer conflictWriter = ConflictWriterType::New(); 
    conflictWriter->SetInput(filter->GetConflictImage()); 
    conflictWriter->SetFileName(argv[2]); 
    conflictWriter->Update(); 
    
    conflictWriter->SetInput(filter->GetForegroundBeliefImage()); 
    conflictWriter->SetFileName("believe.img"); 
    conflictWriter->Update(); 
    conflictWriter->SetInput(filter->GetForegroundPlausibilityImage()); 
    conflictWriter->SetFileName("plausibility.img"); 
    conflictWriter->Update(); 

  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;    
}



