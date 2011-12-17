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
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"

int main(int argc, char** argv)
{
  typedef short TScalar; 
  const int TDimension = 3; 
  typedef itk::Image<TScalar,TDimension> ImageType;
  ImageType::Pointer inputImage; 
  itk::ImageFileReader<ImageType>::Pointer reader;
  
  // Reading it in. 
  try
  {
    reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(argv[1]);
    reader->Update();
    inputImage = reader->GetOutput(); 
    
    ImageType::RegionType region = inputImage->GetLargestPossibleRegion();     
    
    std::cout << "size=" << region.GetSize() << std::endl; 
    if (strcmp(argv[3], "second") == 0)
    {
      region.SetIndex(2, region.GetSize(2)/2-1); 
    }
    region.SetSize(2, region.GetSize(2)/2); 
    
    typedef itk::ImageRegionIterator<ImageType> IteratorType; 
    IteratorType it(inputImage, region); 
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      it.Set(0); 
    }
    
    itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
    writer->SetInput(inputImage); 
    writer->SetFileName(argv[2]); 
    writer->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cout << "ExceptionObject caught !" << std::endl; 
    std::cout << err << std::endl; 
    return EXIT_FAILURE;
  } 
  
  return 0; 
}





