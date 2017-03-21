/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

/*!
 * \file niftkCreatePolyDataFromImage.cxx
 * \page niftkCreatePolyDataFromImage
 * \section niftkCreatePolyDataFromImageSummary niftkCreatePolyDataFromImage creates poly data by thresholding the input target image and applying marching cubes.
 */

#include <itkCommandLineHelper.h>
#include <itkNifTKImageIOFactory.h>
#include <mitkIOUtil.h>

#include <niftkImageToSurfaceFilter.h>
#include <niftkLogHelper.h>

void Usage(char *exec)
{
  niftk::LogHelper::PrintCommandLineHeader(std::cout);
  std::cout << " " << std::endl;
  std::cout << " Creates an polydata object by applying marching cubes to an input image." << std::endl;
  std::cout << " " << std::endl;
  std::cout << " " << exec << " [-i fileInputImage -o fileOutputPolyData] [options]" << std::endl;
  std::cout << " " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << " -i <filename> Input image." << std::endl;
  std::cout << " -o <filename> Output model." << std::endl;
  std::cout << "*** [options] ***" << std::endl << std::endl;
  std::cout << " -t <double> [100.0] threshold value."         << std::endl;
  std::cout << " -e <int> [0] extraction type either (0) vtk Marching cubes or "
    << "(1) corrected marching cubes 33." << std::endl;
  std::cout << " -ir <double> [1.0f] input image downsampling ratio." << std::endl;
  std::cout << " -isType <int> [0] input image smoothing either (0) no smoothing, (1) Gaussian smoothing,"
    << " or (2) Median smoothing." << std::endl;
  std::cout << " -isIter <int> [1] input image smoothing number of iterations." << std::endl;
  std::cout << " -isRad <float> [0.5] input image smoothing radius." << std::endl;
  std::cout << " -ssType <int> [0] output surface smoothing either (0) no smoothing, (1) Taubin smoothing, "
    << "(2) curvature normal smoothing, (3) inverse edge length smoothing, (4) Windowed Sinc smoothing, "
    << "(5) standard VTK smoothing." << std::endl;
  std::cout << " -ssIter <int> [1] output surface smoothing number of iterations." << std::endl;
  std::cout << " -ssRad <float> [0.5] output surface smoothing radius." << std::endl;
  std::cout << " -d <int> [0] surface decimation type either (0) no decimation, (1) Decimate Pro, "
    << "(2) Quadratic VTK, (3) Quadratic, (4) Quadratic Tri, (5) Melax, or (6) Shortest Edge."<< std::endl;
  std::cout << " -tarRed <float> [0.1] polydata reduction ratio."   << std::endl;
  std::cout << " -cleanSurfOn turn on small object removal." << std::endl;
  std::cout << " -cleanT <int> [1000] polygon threshold for small object removal." << std::endl;
}

struct arguments
{
  std::string fileInputImage;
  std::string fileOutputPolyData;
  double threshold = 100.0;
  int extractionType = 0;

  int inputSmoothingType = 0;
  int inputSmoothingIterations = 1;
  float inputSmoothingRadius = 0.5f;
  
  double samplingRatio = 1.0;

  int decimationType = 0;
  float targetReduction = 0.1f;

  int surfSmoothingType = 0;
  int surfSmoothingIterations = 1;
  float surfSmoothingRadius = 0.5f;

  bool useSurfClean = false;
  int  surfCleanThreshold = 1000;
};


//-----------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  itk::NifTKImageIOFactory::Initialize();
 
  // To pass around command line args
  struct arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  for (unsigned int i = 1; i < argc; i++)
  {
    if(strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-Help") == 0 || strcmp(argv[i], "-HELP") == 0 
      || strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--h") == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
    else if (strcmp(argv[i], "-i") == 0)
    {
      args.fileInputImage = argv[++i];
      std::cout << "Set -i=" << args.fileInputImage << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0)
    {
      args.fileOutputPolyData = argv[++i];
      std::cout << "Set -o=" << args.fileOutputPolyData << std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0)
    {
      args.threshold = atof(argv[++i]);
      std::cout << "Set -t=" << args.threshold << std::endl;
    }
    else if(strcmp(argv[i], "-e") == 0)
    {
      args.extractionType = atoi(argv[++i]);
      if (args.extractionType > 1)
      {
        std::cerr << argv[0] << ":\tParameter " << argv[i] << " out of range [0-1]." << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Set -e=" << args.extractionType << std::endl;
    }
    else if(strcmp(argv[i], "-ir") == 0)
    {
      args.samplingRatio = atof(argv[++i]);
      std::cout << "Set -ir=" << args.samplingRatio << std::endl;
    }
    else if(strcmp(argv[i], "-isType") == 0)
    {
      args.inputSmoothingType = atoi(argv[++i]);
      if (args.inputSmoothingType > 2)
      {
        std::cerr << argv[0] << ":\tParameter " << argv[i] << " out of range [0-2]." << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Set -isType=" << args.inputSmoothingType << std::endl;
    }
    else if(strcmp(argv[i], "-isIter") == 0)
    {
      args.inputSmoothingIterations = atoi(argv[++i]);
      std::cout << "Set -isIter=" << args.inputSmoothingIterations << std::endl;
    }
    else if(strcmp(argv[i], "-isRad") == 0)
    {
      args.inputSmoothingRadius = atof(argv[++i]);
      std::cout << "Set -isRad=" << args.inputSmoothingRadius << std::endl;
    }
    else if(strcmp(argv[i], "-ssType") == 0)
    {
      args.surfSmoothingType = atoi(argv[++i]);
      if (args.surfSmoothingType > 5)
      {
        std::cerr << argv[0] << ":\tParameter " << argv[i] << " out of range [0-5]." << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Set -ssType=" << args.surfSmoothingType << std::endl;
    }
    else if(strcmp(argv[i], "-ssIter") == 0)
    {
      args.surfSmoothingIterations = atoi(argv[++i]);
      std::cout << "Set -ssIter=" << args.surfSmoothingIterations << std::endl;
    }
    else if(strcmp(argv[i], "-ssRad") == 0)
    {
      args.surfSmoothingRadius = atof(argv[++i]);
      std::cout << "Set -ssRad=" << args.surfSmoothingRadius << std::endl;
    }
    else if(strcmp(argv[i], "-d") == 0)
    {
      args.decimationType = atoi(argv[++i]);
      if (args.decimationType > 6)
      {
        std::cerr << argv[0] << ":\tParameter " << argv[i] << " out of range [0-6]." << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Set -d=" << args.decimationType << std::endl;
    }
    else if(strcmp(argv[i], "-tarRed") == 0)
    {
      args.targetReduction = atof(argv[++i]);
      std::cout << "Set -tarRed=" << args.targetReduction << std::endl;
    }
    else if (strcmp(argv[i], "--cleanSurf") == 0)
    {
      args.useSurfClean = true;
      std::cout << "Set --cleanSurf = ON" << std::endl;
    }
    else if(strcmp(argv[i], "-cleanT") == 0)
    {
      args.surfCleanThreshold = atoi(argv[++i]);
      std::cout << "Set -cleanT=" << args.surfCleanThreshold << std::endl;
    }
    else 
    {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }      
  }

  if (args.fileInputImage.length() == 0 || args.fileOutputPolyData.length() == 0)
  {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  // load in image
  mitk::Image::Pointer inputImage = mitk::IOUtil::LoadImage(args.fileInputImage);
  if (inputImage.IsNull())
  {
    std::cerr << "Unable to load input image " << args.fileInputImage.c_str() << "." << std::endl;
    return EXIT_FAILURE;
  }

  // Create a mask of the correct dimension
  niftk::ImageToSurfaceFilter::Pointer surfaceFilter = niftk::ImageToSurfaceFilter::New();
  surfaceFilter->SetInput(inputImage);
  surfaceFilter->SetThreshold(args.threshold);

  if (args.extractionType == 0)
  {
    surfaceFilter->SetSurfaceExtractionType(niftk::ImageToSurfaceFilter::StandardExtractor);
  }
  else if (args.extractionType == 1)
  {
    surfaceFilter->SetSurfaceExtractionType(niftk::ImageToSurfaceFilter::EnhancedCPUExtractor);
  }

  if (args.inputSmoothingType == 0)
  {
    surfaceFilter->SetPerformInputSmoothing(false);
  }
  else if (args.inputSmoothingType == 1)
  {
    surfaceFilter->SetPerformInputSmoothing(true);
    surfaceFilter->SetInputSmoothingType(niftk::ImageToSurfaceFilter::GaussianSmoothing);
  }
  else if (args.inputSmoothingType == 2)
  {
    surfaceFilter->SetPerformInputSmoothing(true);
    surfaceFilter->SetInputSmoothingType(niftk::ImageToSurfaceFilter::MedianSmoothing);
  }

  surfaceFilter->SetInputSmoothingIterations(args.inputSmoothingIterations);
  surfaceFilter->SetInputSmoothingRadius(args.inputSmoothingRadius);

  if (args.surfSmoothingType == 0)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(false);
  }
  else if (args.surfSmoothingType == 1)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(true);
    surfaceFilter->SetSurfaceSmoothingType(niftk::ImageToSurfaceFilter::TaubinSmoothing);
  }
  else if (args.surfSmoothingType == 2)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(true);
    surfaceFilter->SetSurfaceSmoothingType(niftk::ImageToSurfaceFilter::CurvatureNormalSmooth);
  }
  else if (args.surfSmoothingType == 3)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(true);
    surfaceFilter->SetSurfaceSmoothingType(niftk::ImageToSurfaceFilter::InverseEdgeLengthSmooth);
  }
  else if (args.surfSmoothingType == 4)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(true);
    surfaceFilter->SetSurfaceSmoothingType(niftk::ImageToSurfaceFilter::WindowedSincSmoothing);
  }
  else if (args.surfSmoothingType == 5)
  {
    surfaceFilter->SetPerformSurfaceSmoothing(true);
    surfaceFilter->SetSurfaceSmoothingType(niftk::ImageToSurfaceFilter::StandardVTKSmoothing);
  }

  surfaceFilter->SetSurfaceSmoothingIterations(args.surfSmoothingIterations);
  surfaceFilter->SetSurfaceSmoothingRadius(args.surfSmoothingRadius);

    if (args.decimationType == 0)
  {
    surfaceFilter->SetPerformSurfaceDecimation(false);
  }
  else if (args.decimationType == 1)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::DecimatePro);
  }
  else if (args.decimationType == 2)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::QuadricVTK);
  }
  else if (args.decimationType == 3)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::Quadric);
  }
  else if (args.decimationType == 4)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::QuadricTri);
  }
  else if (args.decimationType == 5)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::Melax);
  }
  else if (args.decimationType == 6)
  {
    surfaceFilter->SetPerformSurfaceDecimation(true);
    surfaceFilter->SetSurfaceDecimationType(niftk::ImageToSurfaceFilter::ShortestEdge);
  }

  surfaceFilter->SetTargetReduction(args.targetReduction);

  surfaceFilter->SetPerformSurfaceCleaning(args.useSurfClean);
  surfaceFilter->SetSurfaceCleaningThreshold(args.surfCleanThreshold);
  
  surfaceFilter->Update();
  mitk::Surface::Pointer surf = surfaceFilter->GetOutput();

  if (surf.IsNull())
  {
    std::cerr << "Unable to create surface." << std::endl;
    return EXIT_FAILURE;
  }

  bool isSaved = mitk::IOUtil::SaveSurface(surf, args.fileOutputPolyData);
  if (!isSaved)
  {
    std::cerr << "Unable to save surface to file " << args.fileOutputPolyData.c_str() << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
