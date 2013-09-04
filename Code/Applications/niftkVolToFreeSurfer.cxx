/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkContinuousIndex.h>
#include <itkPoint.h>
#include <stdio.h>

/*!
 * \file niftkVolToFreeSurfer.cxx
 * \page niftkVolToFreeSurfer
 * \section niftkVolToFreeSurferSummary Takes a FreeSurfer surface (in ASCII format), and a volume containing (eg.) thickness data, and then for each point in the surface, finds the thickness, either the closest in the neighborhood, or by smoothing and dividing the volume data.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes a FreeSurfer surface (in ASCII format), and a volume containing (eg.) thickness data, and then for each point in the surface, finds the thickness, either the closest in the neighborhood, or by smoothing and dividing the volume data." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -s surfaceFileName -v volumeFileName -o outputFileName" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -s <filename>         Input surface file in ASCII format " << std::endl;
    std::cout << "    -v <filename>         Input volume image in any ITK format " << std::endl;
    std::cout << "    -o <filename>         Output thickness file in ASCII format " << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -b <float>       [0]  Background value in volume image" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "    Choose either" << std::endl;
    std::cout << "      -radius <float>     Search radius to find closest point in volume e.g. 2mm radius" << std::endl;
    std::cout << "      -steps <int>        How many steps to perform over that radius e.g. 5" << std::endl << std::endl;
    std::cout << "    OR" << std::endl;
    std::cout << "      -fwhm  <float>      FWHM for Gaussian smoothing" << std::endl;
    std::cout << "      -mask  <filename>   Binary [0-1] mask image" << std::endl;
    return;
  }

/**
 * \brief Takes mask and target and crops target where mask is non zero.
 */
int main(int argc, char** argv)
{

  const   unsigned int Dimension = 3;
  typedef float        PixelType;

  // Define command line args
  std::string surfaceFile;
  std::string volumeImageName;
  std::string outputFile;
  std::string maskImageName;
  double fwhm = -1;
  double background = 0;
  double radius = -1;
  int    steps = 5;
  bool   useSmoothing = false;
  
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-s") == 0){
      surfaceFile=argv[++i];
      std::cout << "Set -s=" << surfaceFile << std::endl;
    }
    else if(strcmp(argv[i], "-v") == 0){
      volumeImageName=argv[++i];
      std::cout << "Set -v=" << volumeImageName << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      outputFile=argv[++i];
      std::cout << "Set -o=" << outputFile << std::endl;
    }
    else if(strcmp(argv[i], "-mask") == 0){
      maskImageName=argv[++i];
      std::cout << "Set -mask=" << maskImageName << std::endl;
    }    
    else if(strcmp(argv[i], "-fwhm") == 0){
      fwhm=atof(argv[++i]);
      useSmoothing=true;
      std::cout << "Set -fwhm=" << niftk::ConvertToString(fwhm) << std::endl;
    }
    else if(strcmp(argv[i], "-b") == 0){
      background=atof(argv[++i]);
      std::cout << "Set -b=" << niftk::ConvertToString(background) << std::endl;
    }    
    else if(strcmp(argv[i], "-radius") == 0){
      radius=atof(argv[++i]);
      useSmoothing=false;
      std::cout << "Set -radius=" << niftk::ConvertToString(radius) << std::endl;
    }
    else if(strcmp(argv[i], "-steps") == 0){
      steps=atoi(argv[++i]);
      std::cout << "Set -steps=" << niftk::ConvertToString(steps) << std::endl;
    }       
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }    
  }

  // Validate command line args
  if (surfaceFile.length() == 0 || volumeImageName.length() == 0 || outputFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (radius == -1 && fwhm == -1)
    {
      std::cerr << argv[0] << ":\tYou must specify either -radius or -fwhm" << std::endl;
      return EXIT_FAILURE;
    }

  if (useSmoothing && maskImageName.length() < 1)
    {
      std::cerr << argv[0] << ":\tIf you specify -fwhm you must specify a mask image" << std::endl;
      return EXIT_FAILURE;
    }
  
  if (!useSmoothing && radius < 0)
    {
      std::cerr << argv[0] << ":\tRadius must be >= 0" << std::endl;  
      return EXIT_FAILURE;
    }

  if (steps < 1)
    {
      std::cerr << argv[0] << ":\tSteps must be >= 1" << std::endl;
      return EXIT_FAILURE;
    }

  typedef itk::Image< PixelType, Dimension > InputImageType;  
  typedef InputImageType::SpacingType SpacingType;
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::DiscreteGaussianImageFilter<InputImageType, InputImageType> GaussianFilterType;
  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DivideFilterType;
  typedef itk::LinearInterpolateImageFunction< InputImageType, PixelType > LinearInterpolatorType;
  typedef itk::NearestNeighborInterpolateImageFunction< InputImageType, PixelType> NearestNeighbourInterpolatorType;
  
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  InputImageType::Pointer volumeImage = InputImageType::New();
  GaussianFilterType::Pointer gaussianDataFilter = GaussianFilterType::New();
  GaussianFilterType::Pointer gaussianMaskFilter = GaussianFilterType::New();
  DivideFilterType::Pointer divideFilter = DivideFilterType::New();
  LinearInterpolatorType::Pointer linearInterpolator = LinearInterpolatorType::New();
  NearestNeighbourInterpolatorType::Pointer neighbourInterpolator = NearestNeighbourInterpolatorType::New();
  
  FILE* ip;
  if ((ip = fopen(surfaceFile.c_str(), "r")) == NULL)
    {
      std::cerr <<"Failed to open file:" << surfaceFile << " for reading";
      return EXIT_FAILURE;
    }
  
  FILE* op;
  if ((op = fopen(outputFile.c_str(), "w")) == NULL)
    {
      std::cerr <<"Failed to open file:" << outputFile << " for writing";
      return EXIT_FAILURE;
    }

  try 
    { 
      std::cout << "Loading input image:" << volumeImageName << std::endl;
      imageReader->SetFileName(volumeImageName);
      imageReader->Update();
      std::cout << "Done" << std::endl;
      
      if (useSmoothing)
        {
          std::cout << "Loading input mask:" << maskImageName << std::endl;
          maskReader->SetFileName(maskImageName);
          maskReader->Update();
          std::cout << "Done" << std::endl;
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                
  
  if (useSmoothing)
    {
      // http://mathworld.wolfram.com/GaussianFunction.html
      double stdDev = fwhm / (2.0 * sqrt(2.0 * log(2.0)));
      double var = stdDev * stdDev;

      SpacingType variance;
      variance.Fill(var);
      
      std::cout << "Gaussian smoothing data with fwhm=" << niftk::ConvertToString(fwhm) << ", or variance=" << niftk::ConvertToString(var) << std::endl;
      gaussianDataFilter->SetInput(imageReader->GetOutput());
      gaussianDataFilter->SetUseImageSpacing(true);
      gaussianDataFilter->SetVariance(variance);
      gaussianDataFilter->UpdateLargestPossibleRegion();
      
      std::cout << "Gaussian smoothing mask with fwhm=" << niftk::ConvertToString(fwhm) << ", or variance=" << niftk::ConvertToString(var) << std::endl;
      gaussianMaskFilter->SetInput(maskReader->GetOutput());
      gaussianMaskFilter->SetUseImageSpacing(true);
      gaussianMaskFilter->SetVariance(variance);
      gaussianMaskFilter->UpdateLargestPossibleRegion();
      
      std::cout << "Gaussian smoothing ... DONE" << std::endl;
      
      std::cout << "Masking" << std::endl;
      divideFilter->SetInput(0, gaussianDataFilter->GetOutput());
      divideFilter->SetInput(1, gaussianMaskFilter->GetOutput());
      divideFilter->UpdateLargestPossibleRegion();
      std::cout << "Masking ... DONE" << std::endl;
      
      volumeImage = divideFilter->GetOutput();
    }
  else
    {
      volumeImage = imageReader->GetOutput();
    }

  // Now we need to read ASCI file, one point at a time.
  // For each point we:
  //
  // EITHER: Find the closest point in the volume that isn't the background value
  // OR:     Just interpolate the smoothed data. (throw warnings if we find background values).
  //
  // and then write the resultant thickness value, and point location to an output text file.
  
  char lineOfText[256];
  unsigned int lineNumber = 0;
  unsigned int pointNumber = 0;
  unsigned int totalPoints = 1;
  unsigned int unusedInt = 0;
  unsigned int pointsThatHitBackground = 0;
  unsigned int pointsThatWereOutOfBounds = 0;
  float unusedFloat = 0;
  float point[3];
  float dataValue = 0;
  
  linearInterpolator->SetInputImage(volumeImage);
  neighbourInterpolator->SetInputImage(volumeImage);
  
  itk::ContinuousIndex<float, Dimension> continousIndex; 
  itk::Point<float, Dimension> interpolatedPoint;
  itk::Point<float, Dimension> movingPoint;
  
  do 
    {
      if (fgets(lineOfText, 255, ip) != NULL)
        {
          if (lineNumber == 0)
            {
              std::cout << "Skipping first line" << std::endl;
            }
          else if (lineNumber == 1)
            {
              sscanf(lineOfText, "%d %d\n", &totalPoints, &unusedInt);
              std::cout << "There should be " <<  niftk::ConvertToString((int)totalPoints) << " points" << std::endl;
            }
          else
            {
              dataValue = 0;
              
              // Read point.
              sscanf(lineOfText, "%f %f %f %f\n", &point[0], &point[1], &point[2], &unusedFloat);

              // Surface points are in RAS, so we want voxel coordinate
              continousIndex[0] = 128 - point[0];  
              continousIndex[1] = 128 - point[2];
              continousIndex[2] = point[1] + 128;
              
              volumeImage->TransformContinuousIndexToPhysicalPoint(continousIndex, interpolatedPoint);

              if (useSmoothing)
                {
                  dataValue = linearInterpolator->Evaluate(interpolatedPoint);  
                }
              else
                {
                  if (radius == 0)
                    {
                      dataValue = neighbourInterpolator->Evaluate(interpolatedPoint);    
                    }
                  else
                    {
                      double tmpValue = 0;
                      double distance = 0;
                      double valueAtMinDistance = 0;
                      double minDistance = std::numeric_limits<double>::max();
                      
                      for (int z = -steps; z <= steps; z++)
                        {
                          for (int y = -steps; y <= steps; y++)
                            {
                              for (int x = -steps; x <= steps; x++)
                                {
                                  double xoff = x * (radius/(double)steps);
                                  double yoff = y * (radius/(double)steps);
                                  double zoff = z * (radius/(double)steps);
                                  
                                  movingPoint[0] = interpolatedPoint[0] + xoff;
                                  movingPoint[1] = interpolatedPoint[1] + yoff;
                                  movingPoint[2] = interpolatedPoint[2] + zoff;
                                  
                                  tmpValue = neighbourInterpolator->Evaluate(movingPoint);
                                  
                                  if (tmpValue != background)
                                    {
                                      distance = sqrt(xoff*xoff + yoff*yoff + zoff*zoff);
                                      
                                      if (distance < minDistance)
                                        {
                                          minDistance = distance;
                                          valueAtMinDistance = tmpValue;
                                        }
                                    } // end if
                                } // end for x
                            } // end for y
                        } // end for z
                      
                      dataValue = valueAtMinDistance;
                      
                    } // end else
                } // end else
              
              if (dataValue == background)
                {
                  /*
                  std::cout << "Point number " + niftk::ConvertToString((int)pointNumber) \
                    + " at [" + niftk::ConvertToString((double)continousIndex[0]) \
                    + ", " + niftk::ConvertToString((double)continousIndex[1]) \
                    + ", " + niftk::ConvertToString((double)continousIndex[2]) \
                    + "] hit the background");
                    */
                  pointsThatHitBackground++;
                }

              // Write to result to file.
              fprintf(op, "%03d %f %f %f %f\n", pointNumber, point[0], point[1], point[2], dataValue);
              
              pointNumber++;
            }          
          lineNumber++;
        }
      else
        {
          std::cerr <<"Failed to read the line of text";
          return EXIT_FAILURE;
        }
      
    } while (pointNumber < totalPoints);
  
  std::cout << "Actually read " << niftk::ConvertToString((int)pointNumber) << " points and " << niftk::ConvertToString((int)lineNumber) << " lines" << std::endl;

  if (ip != NULL)
    {
      fclose(ip);
    }
  else
    {
      std::cerr <<"Wierd, my input file pointer is NULL, this shouldn't happen, but I'm exiting anyway";
    }
  
  if (op != NULL)
    {
      fclose(op);
    }
  else
    {
      std::cerr <<"Wierd, my output file pointer is NULL, this shouldn't happen, but I'm exiting anyway";
    }

  if (pointsThatHitBackground > 0)
    {
      std::cout << "There were " << niftk::ConvertToString((int)pointsThatHitBackground) << " points that hit the background, so weren't interpolated." << std::endl;
    }
  
  if (pointsThatWereOutOfBounds > 0)
    {
      std::cerr << "There were " << niftk::ConvertToString((int)pointsThatWereOutOfBounds) << " points that were out of bounds, so weren't interpolated.";
    }

  return EXIT_SUCCESS; 
}
