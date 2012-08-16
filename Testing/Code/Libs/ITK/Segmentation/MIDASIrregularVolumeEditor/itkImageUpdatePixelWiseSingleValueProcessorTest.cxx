/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 05:46:51 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7444 $
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
#include "itkImageUpdatePixelWiseSingleValueProcessor.h"

/**
 * Basic tests for itkImageUpdatePixelWiseSingleValueProcessor
 */
int itkImageUpdatePixelWiseSingleValueProcessorTest(int argc, char * argv[])
{

  itk::ImageUpdatePixelWiseSingleValueProcessor<unsigned char, 2>::Pointer processor
    = itk::ImageUpdatePixelWiseSingleValueProcessor<unsigned char, 2>::New();

  typedef itk::Image<unsigned char, 2> ImageType;

  ImageType::Pointer image = itk::Image<unsigned char, 2>::New();
  ImageType::IndexType index;
  ImageType::SizeType size;
  ImageType::RegionType region;

  index.Fill(0);
  size.Fill(2);

  region.SetSize(size);
  region.SetIndex(index);

  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(2);

  index.Fill(0);
  processor->AddToList(index);
  index.Fill(1);
  processor->AddToList(index);

  processor->SetDestinationImage(image);
  processor->SetValue(1);
  processor->SetDebug(true);

  std::cerr << "Calling first redo" << std::endl;

  processor->Redo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking first redo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "1. At index=" << index << ", was expecting 1, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "2. At index=" << index << ", was expecting 1, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling first undo" << std::endl;

  processor->Undo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking first undo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "3. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 2)
  {
    std::cerr << "4. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "Calling second redo" << std::endl;

  processor->Redo();
  image = processor->GetDestinationImage();

  std::cerr << "Checking second redo" << std::endl;

  index.Fill(1);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "5. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
  index.Fill(0);
  if (image->GetPixel(index) != 1)
  {
    std::cerr << "6. At index=" << index << ", was expecting 2, but got:" << image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
