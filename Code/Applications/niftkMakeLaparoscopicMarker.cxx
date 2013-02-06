/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "niftkMakeLaparoscopicMarkerCLP.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"

bool IsOnEvenSquare(const std::vector<int>& vec, const int pixelIndex)
{
  unsigned int chosenIndex = 0;
  for (chosenIndex = 0; chosenIndex < vec.size(); chosenIndex++)
  {
    if (vec[chosenIndex] >= pixelIndex)
    {
      break;
    }
  }
  //std::cout << "Matt, pixelIndex=" << pixelIndex << ", chosenIndex=" << chosenIndex << ", result=" << (chosenIndex % 2 == 0) << std::endl;
  return (chosenIndex % 2 == 0);
}

void CalculateBoundaryArray(const int& pixels, const int& squares, const float& multiplier, std::vector<int>& vec)
{
  float a = 0;
  float b = multiplier;

  float accumulator = 1;
  for (int i = 1; i < squares; i++)
  {
    accumulator += pow(multiplier, i);
  }
  a = (float)pixels / accumulator;

  vec.clear();

  // First number is always
  vec.push_back(a);

  // Numbers in between are a geometric progression a + a*b + a*b*b + a*b*b*b
  for (int i = 1; i < squares-1; i++)
  {
    vec.push_back(vec[vec.size()-1]
        + (int)(a*pow(multiplier,i)+0.5));
  }

  // Last number is always the number of pixels.
  vec.push_back(pixels);

  std::cout << "Pixels=" << pixels << ", squares=" << squares << ", multiplier=" << multiplier << ", so a=" << a << ", b=" << b << std::endl;
  for (unsigned int i = 0; i < vec.size(); i++)
  {
    std::cout << vec[i] << ", ";
  }
  std::cout << std::endl;
}

/**
 * \brief Generates a 2D image with a calibration pattern.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  std::vector<int> xboundaries;
  std::vector<int> yboundaries;

  CalculateBoundaryArray(width, squaresAlongWidth, multiplierWidth, xboundaries);
  CalculateBoundaryArray(height, squaresAlongHeight, multiplierHeight, yboundaries);

  typedef itk::Image<unsigned char, 2>    ImageType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef ImageType::RegionType           RegionType;
  typedef ImageType::SizeType             SizeType;
  typedef ImageType::IndexType            IndexType;

  SizeType size;
  size[0] = width;
  size[1] = height;
  IndexType imageIndex;
  imageIndex[0] = 0;
  imageIndex[1] = 0;
  RegionType region;
  region.SetSize(size);
  region.SetIndex(imageIndex);

  ImageType::Pointer image = ImageType::New();
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      imageIndex[0] = x;
      imageIndex[1] = y;

      bool xEven = IsOnEvenSquare(xboundaries, x);
      bool yEven = IsOnEvenSquare(yboundaries, y);

      if ((xEven && !yEven) || (yEven && !xEven))
      {
        image->SetPixel(imageIndex, 255);
      }
      else
      {
        image->SetPixel(imageIndex, 0);
      }
    }
  }

  ImageWriterType::Pointer writer = ImageWriterType::New();
  writer->SetInput(image);
  writer->SetFileName(output);
  writer->Update();

  return EXIT_SUCCESS;
}
