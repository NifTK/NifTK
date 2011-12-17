/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author:  $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMedianImageFilter.h"


int main(int argc, char* argv[])
{  
  const unsigned int Dimension = 3; 
  typedef short PixelType; 
  typedef itk::Image<PixelType, Dimension> InputImageType;   
  typedef itk::ImageFileReader<InputImageType> InputImageReaderType;
  typedef itk::ImageFileWriter<InputImageType> OutputImageWriterType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  
  imageReader->SetFileName(argv[1]);
  imageWriter->SetFileName(argv[2]);
  unsigned long radius = atoi(argv[3]); 
  InputImageType::SizeType size; 
  
  size.Fill(radius); 
  
  typedef itk::MedianImageFilter<InputImageType, InputImageType> ImageFilterType; 
  ImageFilterType::Pointer imageFilter = ImageFilterType::New();
  
  imageFilter->SetRadius(size); 
  imageFilter->SetInput(imageReader->GetOutput()); 
  imageWriter->SetInput(imageFilter->GetOutput()); 
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  return EXIT_SUCCESS;
}
