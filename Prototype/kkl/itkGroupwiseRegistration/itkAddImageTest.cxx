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

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAddImageFilter.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimension > FixedImageType;
  typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
  typedef itk::ImageFileWriter< FixedImageType > FixedImageWriterType;
  
  // Read in the fixed images. 
  FixedImageReaderType::Pointer imageReader1 = FixedImageReaderType::New();
  FixedImageReaderType::Pointer imageReader2 = FixedImageReaderType::New();
  FixedImageWriterType::Pointer writer = FixedImageWriterType::New(); 

  imageReader1->SetFileName(argv[1]); 
  imageReader2->SetFileName(argv[2]); 
  writer->SetFileName(argv[3]); 
  
  typedef itk::AddImageFilter<FixedImageType, FixedImageType, FixedImageType> AddImageFilterType; 
  AddImageFilterType::Pointer filter = AddImageFilterType::New(); 
  
  filter->SetInput1(imageReader1->GetOutput()); 
  filter->SetInput2(imageReader2->GetOutput()); 
  writer->SetInput(filter->GetOutput()); 
  writer->Update(); 
  
  return 0;   
}
