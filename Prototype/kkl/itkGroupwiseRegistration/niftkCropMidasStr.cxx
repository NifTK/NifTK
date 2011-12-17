/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: kkl $

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
#include <fstream>
#include <cstdlib>

int main(int argc, char** argv)
{
  const unsigned int NDimensions = 3; 
  const char* inputFilename = argv[1]; 
  const char* outputFilename = argv[2];
  // Size to be cropped to. 
  int imageCropSize[NDimensions];
  imageCropSize[0] = atoi(argv[3]);    
  imageCropSize[1] = atoi(argv[4]); 
  imageCropSize[2] = atoi(argv[5]); 
  std::ifstream inputStream(inputFilename); 
  int size[NDimensions]; 
  int totalSize = 1; 
  int origin[NDimensions]; 
    
  // Image size. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    inputStream.read(reinterpret_cast<char*>(&size[i]), sizeof(size[i])); 
    std::cout << "size=" << size[i] << std::endl; 
    totalSize *= size[i]; 
  }
  
  // Jacobian values. 
  float* jacobian = new float[totalSize]; 
  for (int z = 0; z < size[2]; z++)
  {
    for (int y = 0; y < size[1]; y++)
    {
      for (int x = 0; x < size[0]; x++)
      {
        int index = z*size[0]*size[1] + y*size[0] + x; 
        inputStream.read(reinterpret_cast<char*>(&jacobian[index]), sizeof(jacobian[index])); 
      }
    }
  }
  
  // Image origin. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    inputStream.read(reinterpret_cast<char*>(&origin[i]), sizeof(origin[i])); 
    std::cout << "origin=" << origin[i] << std::endl; 
  }
  
  inputStream.close(); 
  
  std::ofstream outputStream(outputFilename); 
  
  // Jacobian values. 
  int totalCroppedSize = 1;
  int startingPos[NDimensions]; 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    startingPos[i] = (size[i]-imageCropSize[i])/2; 
    totalCroppedSize *= imageCropSize[i]; 
    std::cout << "imageCropSize=" << imageCropSize[i] << std::endl; 
  }
  std::cout << "totalCroppedSize=" << totalCroppedSize << std::endl; 
  
  // Image size. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    outputStream.write(reinterpret_cast<char*>(&imageCropSize[i]), sizeof(imageCropSize[i])); 
  }
  
  // Jaocbian. 
  int count = 0; 
  for (int z = startingPos[2]; z < startingPos[2]+imageCropSize[2]; z++)
  {
    for (int y = startingPos[1]; y < startingPos[1]+imageCropSize[1]; y++)
    {
      for (int x = startingPos[0]; x < startingPos[0]+imageCropSize[0]; x++)
      {
        int index = z*size[0]*size[1] + y*size[0] + x; 
        outputStream.write(reinterpret_cast<char*>(&jacobian[index]), sizeof(jacobian[index])); 
        count++; 
      }
    }
  }
  std::cout << "count=" << count << std::endl; 
  
  // Image origin. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    outputStream.write(reinterpret_cast<char*>(&origin[i]), sizeof(origin[i])); 
    std::cout << "origin=" << origin[i] << std::endl; 
  }
  
  delete[] jacobian; 
  return 0; 
  
}














