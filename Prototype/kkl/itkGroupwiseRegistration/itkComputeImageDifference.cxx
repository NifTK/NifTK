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
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"

int main(int argc, char* argv[])
{
  typedef short PixelType; 
  const unsigned int Dimensions = 3; 
  typedef itk::Image< PixelType, Dimensions > ImageType; 
  typedef itk::ImageFileReader< ImageType > ImageReaderType; 
  
  ImageReaderType::Pointer reader1 = ImageReaderType::New(); 
  ImageReaderType::Pointer reader2 = ImageReaderType::New(); 
  reader1->SetFileName(argv[1]); 
  reader1->Update(); 
  reader2->SetFileName(argv[2]); 
  reader2->Update(); 
  
  typedef itk::ImageRegionConstIterator< ImageType > ImageRegionConstIteratorType; 
  ImageRegionConstIteratorType it1(reader1->GetOutput(), reader1->GetOutput()->GetLargestPossibleRegion());
  ImageRegionConstIteratorType it2(reader2->GetOutput(), reader2->GetOutput()->GetLargestPossibleRegion());
  
  double sum = 0; 
  
  for (it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2)
  {
    sum += (it1.Get()-it2.Get()); 
  }
  
  std::cout << sum << std::endl; 

  return 0; 
}



