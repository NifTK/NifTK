/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkContourExtractor2DImageFilter.h>
#include <itkOrthogonalContourExtractor2DImageFilter.h>
#include <itkPolyLineParametricPath.h>

/*!
 * \file niftkContourExtractor2DImageFilter.cxx
 * \page niftkContourExtractor2DImageFilter
 * \brief niftkContourExtractor2DImageFilter - Runs either the ITK ContourExtractor2DImageFilter or NifTK's modified version OrthogonalContourExtractor2DImageFilter, taking 2D images and performing marching squares contour extraction."
 *
 * See also <a href="http://www.itk.org/Doxygen316/html/classitk_1_1ContourExtractor2DImageFilter.html">the ITK manual page</a>.
 *
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs ITK ContourExtractor2DImageFilter or NifTK's modified version OrthogonalContourExtractor2DImageFilter, taking 2D images and performing marching squares contour extraction." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName1 [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image. " << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -t    threshold         Threshold value." << std::endl;
    std::cout << "    -o                      Do orthogonal extraction using NifTK OrthogonalContourExtractor2DImageFilter." << std::endl;
  }

struct arguments
{
  std::string inputImage;
  float       threshold;
  bool        doOrthogonal;
};

template <int Dimension, class PixelType>
int DoMain(arguments args)
{
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::PolyLineParametricPath<Dimension> PathType;
  typedef typename itk::ContourExtractor2DImageFilter<InputImageType> ImageFilterType;
  typedef typename itk::OrthogonalContourExtractor2DImageFilter<InputImageType> OrthogonalImageFilterType;

  // A bit basic ... but it works for now.

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  typename ImageFilterType::Pointer filter = ImageFilterType::New();
  filter->SetInput(imageReader->GetOutput());
  filter->SetContourValue(args.threshold);

  typename OrthogonalImageFilterType::Pointer orthoFilter = OrthogonalImageFilterType::New();
  orthoFilter->SetInput(imageReader->GetOutput());
  orthoFilter->SetContourValue(args.threshold);

  try
  {
    int numberOfOutputs = 0;

    if (args.doOrthogonal)
    {
      orthoFilter->Update();
      numberOfOutputs = orthoFilter->GetNumberOfOutputs();
    }
    else
    {
      filter->Update();
      numberOfOutputs = filter->GetNumberOfOutputs();
    }

    for (int i = 0; i < numberOfOutputs; i++)
    {
      typename PathType::Pointer path;

      if (args.doOrthogonal)
      {
        path = orthoFilter->GetOutput(i);
      }
      else
      {
        path = filter->GetOutput(i);
      }

      const typename PathType::VertexListType* list = path->GetVertexList();
      typename PathType::VertexType vertex;

      for (unsigned long int j = 0; j < list->Size(); j++)
      {
        vertex = list->ElementAt(j);
        std::cerr << i << "," << j  << ": " << vertex << std::endl;
      }
    }
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * \brief Takes image1 and image2 and adds them together
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.threshold = 0;
  args.doOrthogonal = false;

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage<< std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      args.threshold=atof(argv[++i]);
      std::cout << "Set -t=" << args.threshold<< std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.doOrthogonal=true;
      std::cout << "Set -o=" << niftk::ConvertToString(args.doOrthogonal) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validate command line args
  if (args.inputImage.length() == 0 )
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<2, unsigned char>(args);
      break;
    case itk::ImageIOBase::CHAR:
      result = DoMain<2, char>(args);
      break;
    case itk::ImageIOBase::USHORT:
      result = DoMain<2, unsigned short>(args);
      break;
    case itk::ImageIOBase::SHORT:
      result = DoMain<2, short>(args);
      break;
    case itk::ImageIOBase::UINT:
      result = DoMain<2, unsigned int>(args);
      break;
    case itk::ImageIOBase::INT:
      result = DoMain<2, int>(args);
      break;
    case itk::ImageIOBase::ULONG:
      result = DoMain<2, unsigned long>(args);
      break;
    case itk::ImageIOBase::LONG:
      result = DoMain<2, long>(args);
      break;
    case itk::ImageIOBase::FLOAT:
      result = DoMain<2, float>(args);
      break;
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<2, double>(args);
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
