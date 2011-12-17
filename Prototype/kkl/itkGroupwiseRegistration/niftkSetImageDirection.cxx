/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-07-22 10:11:40 +0100 (Thu, 22 Jul 2010) $
 Revision          : $Revision: 3539 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"


int main(int argc, char* argv[]) 
{
  typedef short PixelType; 
  const unsigned int Dimensions = 3; 
  typedef itk::Image< PixelType, Dimensions > ImageType; 
  typedef itk::ImageFileReader< ImageType > ImageReaderType;
  typedef itk::ImageFileWriter< ImageType > ImageWriterType;
   
  const char* inputImage1Name = argv[1]; 
  const char* inputImage2Name = argv[2]; 
  const char* outputImageName = argv[3]; 

  try
  {
    ImageReaderType::Pointer reader1 = ImageReaderType::New(); 
    ImageReaderType::Pointer reader2 = ImageReaderType::New(); 
    reader1->SetFileName(inputImage1Name); 
    reader1->Update(); 
    
    std::cout << "Current direction=" << reader1->GetOutput()->GetDirection() << std::endl; 
    
    ImageType::DirectionType direction; 
    
    direction.SetIdentity(); 
    if (strlen(inputImage2Name) > 0)
    {
      reader2->SetFileName(inputImage2Name); 
      reader2->Update(); 
      direction = reader2->GetOutput()->GetDirection(); 
    }
    std::cout << "Direction to set =" << direction << std::endl; 
    
    reader1->GetOutput()->SetDirection(direction); 
    
    ImageWriterType::Pointer writer = ImageWriterType::New(); 
    writer->SetInput(reader1->GetOutput()); 
    writer->SetFileName(outputImageName); 
    writer->Update(); 
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cout << "Failed: " << exceptionObject << std::endl;
    return 1; 
  }
  return 0; 
}
  

  