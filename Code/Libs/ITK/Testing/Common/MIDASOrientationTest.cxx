/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <cstdlib>
#include <itkImageFileReader.h>
#include <itkImage.h>

/**
 * @brief Tests ITK Loading DRC Analyze files ok.
 *
 * Note: Only passes when NIFTK_DRC_ANALYZE env var is ON.
 *
 * As of 4121, NifTK file readers were moved out of the
 * Code/Gui/MITK area, and into niftkITK library. Then,
 * there are two mechanisms to load them. In MITK there is
 * a Service Registry based approach: see niftkCore/IO,
 * mitkNifTKCoreObjectFactory and unit tests for that module.
 * In "plain ITK", there is the FileReader approach, which works
 * via ITK's Object Factories: see niftkITK/IO/itkNifTKImageIOFactory.
 *
 * These must be kept in synch.
 */
int MIDASOrientationTest(int argc, char * argv[])
{

  if (argc != 5)
    {
      std::cerr << "Usage: AnalyzeOrientationTest filename nx ny nz" << std::endl;
      return EXIT_FAILURE;
    }

  // Plain old ITK. Load image, check size to crudely and somewhat incompletely verify orientation.
  typedef itk::Image<int, 3>              ImageType;
  typedef ImageType::SizeType             SizeType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  ImageReaderType::Pointer reader = ImageReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  ImageType* image = reader->GetOutput();
  SizeType actualSize = image->GetLargestPossibleRegion().GetSize();
  SizeType expectedSize;
  expectedSize[0] = atoi(argv[2]);
  expectedSize[1] = atoi(argv[3]);
  expectedSize[2] = atoi(argv[4]);

  for (int i = 0; i < 3; i++)
  {
    if (actualSize[i] != expectedSize[i])
    {
      std::cerr << "Dimension " << i << ", expected = " << expectedSize[i] << ", actual = " << actualSize[0];
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

