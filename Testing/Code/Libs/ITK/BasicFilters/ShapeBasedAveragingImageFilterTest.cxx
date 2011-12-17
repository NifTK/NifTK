/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

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
#include "itkShapeBasedAveragingImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/**
 * Basic tests for ShapeBasedAveragingImageFilter
 */
int ShapeBasedAveragingImageFilterTest(int argc, char * argv[])
{

  // Declare the types of the images
  const unsigned int Dimension = 2;
  typedef unsigned char PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ShapeBasedAveragingImageFilter<ImageType, ImageType> FilterType;
  typedef itk::ImageFileReader<ImageType> ReaderType; 
  typedef itk::ImageFileWriter<ImageType> WriterType; 
  
  try
  {
    ReaderType::Pointer reader1 = ReaderType::New(); 
    reader1->SetFileName(argv[1]); 
    reader1->Update(); 
    ReaderType::Pointer reader2 = ReaderType::New(); 
    reader2->SetFileName(argv[2]); 
    reader2->Update(); 
    
    FilterType::Pointer filter = FilterType::New(); 
    filter->SetInput(0, reader1->GetOutput()); 
    filter->SetInput(1, reader2->GetOutput()); 
    filter->Update(); 
    
    WriterType::Pointer writer = WriterType::New(); 
    writer->SetInput(filter->GetOutput()); 
    writer->SetFileName(argv[3]); 
    writer->Update(); 
    
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;    
}



