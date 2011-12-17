/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-23 12:27:31 +0100 (Mon, 23 May 2011) $
 Revision          : $Revision: 6241 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iostream>
#include <fstream>
#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>

int main(int argc, char** argv)
{
  const char* inputFilename = argv[1]; 
  const char* outputFilename = argv[2];
  
  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  
  ImageReaderType::Pointer reader = ImageReaderType::New(); 
  ImageType::SizeType regionSize; 
  
  reader->SetFileName(inputFilename); 
  reader->Update(); 
  regionSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize(); 
  std::cout << "regionSize=" << regionSize << std::endl; 
  
  std::ofstream outputStream(outputFilename); 
  
  // Image size. 
  for (unsigned int i = 0; i < Dimension; i++)
  {
    int temp = regionSize[i]; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  
  // Jaocbian. 
  std::cout << "Writing Jacobian..." << std::endl; 
  //for (int z = (int) regionSize[2]-1; z >= 0; z--)
  for (unsigned int z = 0; z < regionSize[2]; z++)
  {
    for (unsigned int y = 0; y < regionSize[1]; y++)
    {
      for (unsigned int x = 0; x < regionSize[0]; x++)
      {
        ImageType::IndexType index; 
        index[0] = x; 
        index[1] = y; 
        index[2] = z; 
        float temp = reader->GetOutput()->GetPixel(index); 
        outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
      }
    }
  }
  
  std::cout << "Writing origin..." << std::endl; 
  // Image origin. 
  for (unsigned i = 0; i < Dimension; i++)
  {
    int temp = 0; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  
  outputStream.close(); 
  return 0; 
}


















  
  
