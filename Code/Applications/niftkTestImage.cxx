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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkContinuousIndex.h>
#include <itkExtractImageFilter.h>

/*!
 * \file niftkTestImage.cxx
 * \page niftkTestImage
 * \section niftkTestImageSummary Generates a 3D test image, either a binary cuboid, binary ellipsoid, grid, or test card type pattern.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Generates a 3D test image, either a binary cuboid, binary ellipsoid, grid, or test card type pattern." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " <mandatory> [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -nx  <int>   128          Number of voxels in x dimension" << std::endl;
    std::cout << "    -ny  <int>   128          Number of voxels in y dimension" << std::endl;
    std::cout << "    -nz  <int>   128          Number of voxels in z dimension" << std::endl;
    std::cout << "    -vx  <float> 1.0          Size of voxels in x dimension" << std::endl;
    std::cout << "    -vy  <float> 1.0          Size of voxels in y dimension" << std::endl;
    std::cout << "    -vz  <float> 1.0          Size of voxels in z dimension" << std::endl;
    std::cout << "    -ox  <float> 0.0          X origin" << std::endl;    
    std::cout << "    -oy  <float> 0.0          Y origin" << std::endl;
    std::cout << "    -oz  <float> 0.0          Z origin" << std::endl;
    std::cout << "    -o   <filename>           Ouput filename" << std::endl;
    std::cout << "    -bv  <int>   0            Background value" << std::endl;
    std::cout << "    -mode [int]  0            Selects the mode of operation. See further parameters below." << std::endl;
    std::cout << "                              0 = generate cuboid" << std::endl;
    std::cout << "                              1 = generate ellipse" << std::endl;
    std::cout << "                              2 = generate grid" << std::endl;
    std::cout << "                              3 = generate testcard image" << std::endl;
    std::cout << "                              4 - generate distance from centre image" << std::endl;
    std::cout << "                              5 - generate increasing voxel number as intensity value" << std::endl;
    std::cout << "                              6 - make calibration chessboard" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;  
    std::cout << "    -dir a b c d e f g h i    Direction matrix" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "For mode:" << std::endl;
    std::cout << "  0" << std::endl;
    std::cout << "      -xr  <float> 20.0         Distance from centre of cuboid to edge on x axis" << std::endl;
    std::cout << "      -yr  <float> 20.0         Distance from centre of cuboid to edge on y axis" << std::endl;
    std::cout << "      -zr  <float> 20.0         Distance from centre of cuboid to edge on z axis" << std::endl;
    std::cout << "      -fv  <int>   1            Foreground value" << std::endl;
    std::cout << "  1" << std::endl;
    std::cout << "      -xr  <float> 20.0         X radius" << std::endl;
    std::cout << "      -yr  <float> 20.0         Y radius" << std::endl;
    std::cout << "      -zr  <float> 20.0         Z radius" << std::endl;
    std::cout << "      -fv  <int>   1            Foreground value" << std::endl;
    std::cout << "  2" << std::endl;
    std::cout << "      -xs <int>    2            X step size in pixels" << std::endl;
    std::cout << "      -ys <int>    2            Y step size in pixels" << std::endl;
    std::cout << "      -zs <int>    2            Z step size in pixels" << std::endl;
    std::cout << "      -fv <int>    1            Foreground value" << std::endl;
    std::cout << "  3" << std::endl;
    std::cout << "      none" << std::endl;
    std::cout << "  4" << std::endl;
    std::cout << "      none" << std::endl;
    std::cout << "  5" << std::endl;
    std::cout << "      none" << std::endl;
    std::cout << "  6" << std::endl;
    std::cout << "      none" << std::endl;
  }

/**
 * \brief Creates a test image, either ellipse or cuboid.
 */
int main(int argc, char** argv)
{
  
  const    unsigned int    Dimension = 3;
  typedef  float           ScalarType;

  // Define command line params
  std::string outputImage;
  int nx=128;
  int ny=128;
  int nz=128;
  int backgroundValue=0;
  int foregroundValue=1;
  double xdim=1.0;
  double ydim=1.0;
  double zdim=1.0;
  double xorigin=0;
  double yorigin=0;
  double zorigin=0;
  double xradius=20;
  double yradius=20;
  double zradius=20;
  float direction[9];
  bool userSuppliedDirection=false;
  int mode=0;
  int xstep=2;
  int ystep=2;
  int zstep=2;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-o") == 0){
      outputImage=argv[++i];
      std::cout << "Set -o=" << outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-nx") == 0){
      nx=atoi(argv[++i]);
      if (nx <= 0){
	std::cerr << "Error: nx must be an integer value above 0" << std::endl;
	return EXIT_FAILURE;
	}
      std::cout << "Set -nx=" << niftk::ConvertToString(nx) << std::endl;
    }
    else if(strcmp(argv[i], "-ny") == 0){
      ny=atoi(argv[++i]);
      if (ny <= 0){
	std::cerr << "Error: ny must be an integer value above 0" << std::endl;
	return EXIT_FAILURE;
	}
      std::cout << "Set -ny=" << niftk::ConvertToString(ny) << std::endl;
    }    
    else if(strcmp(argv[i], "-nz") == 0){
      nz=atoi(argv[++i]);
      if (nz <= 0){
	std::cerr << "Error: nz must be an integer value above 0" << std::endl;
	return EXIT_FAILURE;
	}
      std::cout << "Set -nz=" << niftk::ConvertToString(nz) << std::endl;
    }    
    else if(strcmp(argv[i], "-bv") == 0){
      backgroundValue=atoi(argv[++i]);
      std::cout << "Set -bv=" << niftk::ConvertToString(backgroundValue) << std::endl;
    }    
    else if(strcmp(argv[i], "-fv") == 0){
      foregroundValue=atoi(argv[++i]);
      std::cout << "Set -fv=" << niftk::ConvertToString(foregroundValue) << std::endl;
    }    
    else if(strcmp(argv[i], "-vx") == 0){
      xdim=atof(argv[++i]);
      std::cout << "Set -vx=" << niftk::ConvertToString(xdim) << std::endl;
    }
    else if(strcmp(argv[i], "-vy") == 0){
      ydim=atof(argv[++i]);
      std::cout << "Set -vy=" << niftk::ConvertToString(ydim) << std::endl;
    }
    else if(strcmp(argv[i], "-vz") == 0){
      zdim=atof(argv[++i]);
      std::cout << "Set -vz=" << niftk::ConvertToString(zdim) << std::endl;
    }
    else if(strcmp(argv[i], "-ox") == 0){
      xorigin=atof(argv[++i]);
      std::cout << "Set -ox=" << niftk::ConvertToString(xorigin) << std::endl;
    }
    else if(strcmp(argv[i], "-oy") == 0){
      yorigin=atof(argv[++i]);
      std::cout << "Set -oy=" << niftk::ConvertToString(yorigin) << std::endl;
    }
    else if(strcmp(argv[i], "-oz") == 0){
      zorigin=atof(argv[++i]);
      std::cout << "Set -oz=" << niftk::ConvertToString(zorigin) << std::endl;
    }
    else if(strcmp(argv[i], "-xr") == 0){
      xradius=atof(argv[++i]);
      std::cout << "Set -xr=" << niftk::ConvertToString(xradius) << std::endl;
    }
    else if(strcmp(argv[i], "-yr") == 0){
      yradius=atof(argv[++i]);
      std::cout << "Set -yr=" << niftk::ConvertToString(yradius) << std::endl;
    }
    else if(strcmp(argv[i], "-zr") == 0){
      zradius=atof(argv[++i]);
      std::cout << "Set -zr=" << niftk::ConvertToString(zradius) << std::endl;
    }
    else if (strcmp(argv[i], "-dir") == 0){
      for (unsigned int j = 0; j < Dimension*Dimension; j++)
      {
        direction[j] = atof(argv[++i]);
        std::cout << "Set -direction[j]=" << niftk::ConvertToString(direction[j]) << std::endl;
      }
      userSuppliedDirection=true;
    }
    else if(strcmp(argv[i], "-xs") == 0){
      xstep=atoi(argv[++i]);
      std::cout << "Set -xs=" << niftk::ConvertToString(xstep) << std::endl;
    }
    else if(strcmp(argv[i], "-ys") == 0){
      ystep=atoi(argv[++i]);
      std::cout << "Set -ys=" << niftk::ConvertToString(ystep) << std::endl;
    }
    else if(strcmp(argv[i], "-zs") == 0){
      zstep=atoi(argv[++i]);
      std::cout << "Set -zs=" << niftk::ConvertToString(zstep) << std::endl;
    }
    else if(strcmp(argv[i], "-mode") == 0){
      mode=atoi(argv[++i]);
      std::cout << "Set -mode=" << niftk::ConvertToString(mode) << std::endl;
    }

    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  if (mode < 0 || mode > 6)
  {
    std::cerr << "Invalid mode" << std::endl;
    return EXIT_FAILURE;
  }

 // Validate command line args
  if (outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  typedef itk::Image<ScalarType, Dimension> ImageType;
  typedef itk::ImageRegion<Dimension>       ImageRegionType;
  typedef ImageType::IndexType              ImageIndexType;
  typedef ImageType::SizeType               ImageSizeType;
  typedef ImageType::SpacingType            ImageSpacingType;
  typedef ImageType::PointType              ImageOriginType;
  typedef ImageType::DirectionType          ImageDirectionType;
  typedef itk::ContinuousIndex<float, Dimension> ContinuousIndexType;
  typedef itk::Point<float, Dimension>           PointType;
  
  ImageType::Pointer testImage = ImageType::New();
  ImageRegionType region;

  ImageIndexType index;
  index[0] = 0;
  index[1] = 0;
  index[2] = 0;
  ImageSizeType size;
  size[0] = nx;
  size[1] = ny;
  size[2] = nz;
  ImageSpacingType spacing;
  spacing[0] = xdim;
  spacing[1] = ydim;
  spacing[2] = zdim;
  ImageOriginType origin;
  origin[0] = xorigin;
  origin[1] = yorigin;
  origin[2] = zorigin;
  
  ImageDirectionType directionType;
  if (userSuppliedDirection)
  {
    for (unsigned int y=0; y < Dimension; y++)
    {
      for (unsigned int x=0; x < Dimension; x++)
      {
        int counter = y*Dimension + x;
        directionType[y][x] = direction[counter];
      }
    }
  }
  else
  {
    directionType.SetIdentity();
  }
  region.SetIndex(index);
  region.SetSize(size);
  testImage->SetSpacing(spacing);
  testImage->SetOrigin(origin);
  testImage->SetRegions(region);
  testImage->SetDirection(directionType);
  testImage->Allocate();
  testImage->FillBuffer(backgroundValue);

  ContinuousIndexType middleOfImage;
  middleOfImage[0] = (nx - 1)/2.0;
  middleOfImage[1] = (ny - 1)/2.0;
  middleOfImage[2] = (nz - 1)/2.0;

  if (mode == 0 || mode == 1)
  {
    ImageOriginType radius;
    radius[0] = xradius;
    radius[1] = yradius;
    radius[2] = zradius;

    double xdist;
    double ydist;
    double zdist;

    for (int x = 0; x < nx; x++)
      {
        for (int y = 0; y < ny; y++)
          {
            for (int z = 0; z < nz; z++)
              {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                xdist = (x - middleOfImage[0])*xdim;
                ydist = (y - middleOfImage[1])*ydim;
                zdist = (z - middleOfImage[2])*zdim;

                if (mode == 0)
                  {
                    if (fabs(xdist) < radius[0] && fabs(ydist) < radius[1] && fabs(zdist) < radius[2])
                      {
                        testImage->SetPixel(index, foregroundValue);
                      }
                    else
                      {
                        testImage->SetPixel(index, backgroundValue);
                      }

                  }
                else
                  {
                    if (  (xdist*xdist)/(radius[0]*radius[0])
                        + (ydist*ydist)/(radius[1]*radius[1])
                        + (zdist*zdist)/(radius[2]*radius[2])  < 1)
                      {
                        testImage->SetPixel(index, foregroundValue);
                      }
                    else
                      {
                        testImage->SetPixel(index, backgroundValue);
                      }
                  }
              }
          }
      }
  }
  else if (mode == 2)
  {
    for (int x = 0; x < nx; x += xstep)
      {
        for (int y = 0; y < ny; y += ystep)
          {
            for (int z = 0; z < nz; z += zstep)
              {

                index[0] = x;
                index[1] = y;
                index[2] = z;

                testImage->SetPixel(index, foregroundValue);
              }
          }
      }
  }
  else if (mode == 3 || mode == 4)
  {
    PointType middleOfImageInMillimetres;
    PointType voxelLocationInMillimetres;
    double distance = 0;

    testImage->TransformContinuousIndexToPhysicalPoint(middleOfImage, middleOfImageInMillimetres);

    for (int x = 0; x < nx; x++)
      {
        for (int y = 0; y < ny; y++)
          {
            for (int z = 0; z < nz; z++)
              {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if (mode == 3)
                {
                  if (x == 0 || y == 0 || z == 0 || x == nx-1 || y == ny -1 || z == nz-1)
                  {
                    testImage->SetPixel(index, nx*ny*nz);
                  }
                  else if (x == 1|| y == 1 || z == 1 || x == nx-2 || y == ny -2 || z == nz-2)
                  {
                    testImage->SetPixel(index, backgroundValue);
                  }
                  else
                  {
                    testImage->SetPixel(index, x*y*z);
                  }
                }
                else
                {
                  testImage->TransformIndexToPhysicalPoint(index, voxelLocationInMillimetres);
                  distance = voxelLocationInMillimetres.EuclideanDistanceTo(middleOfImageInMillimetres);
                  testImage->SetPixel(index, distance);
                }
              }
          }
      }
  }
  else if (mode == 5)
  {
    unsigned long int voxelCounter = 0;

    for (int z = 0; z < nz; z++)
      {
        for (int y = 0; y < ny; y++)
          {
            for (int x = 0; x < nx; x++)
              {

                index[0] = x;
                index[1] = y;
                index[2] = z;

                testImage->SetPixel(index, voxelCounter);
                voxelCounter++;
              }
          }
      }
  }
  else if (mode == 6)
  {
    for (int z = 0; z < nz; z++)
      {
        for (int y = 0; y < ny; y++)
          {
            for (int x = 0; x < nx; x++)
              {

                index[0] = x;
                index[1] = y;
                index[2] = z;


                if ( x%2 == 0 && y%2 == 0
                     || x%2 == 1 && y%2 == 1)
                {
                  testImage->SetPixel(index, 255);
                }
              }
          }
      }
  }

  if (mode != 6)
  {
    typedef itk::ImageFileWriter< ImageType  > ImageWriterType;
    ImageWriterType::Pointer writer = ImageWriterType::New();
    writer->SetFileName(outputImage);
    writer->SetInput(testImage);
    writer->Update();
  }
  else
  {
    typedef itk::Image<ScalarType, 2> OutputImageType;
    typedef itk::ExtractImageFilter<ImageType, OutputImageType> ExtractImageFilterType;
    typedef itk::ImageFileWriter< OutputImageType  > OutputImageWriterType;

    ImageType::SizeType outputSize;
    ImageType::RegionType outputRegion;

    outputRegion = testImage->GetLargestPossibleRegion();
    outputSize = outputRegion.GetSize();
    outputSize[2] = 0;
    outputRegion.SetSize(outputSize);

    ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();
    filter->SetInput(testImage);
    filter->SetExtractionRegion(outputRegion);
    filter->SetDirectionCollapseToIdentity();

    OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    imageWriter->SetFileName(outputImage);
    imageWriter->SetInput(filter->GetOutput());
    imageWriter->Update();
  }
  
  return 0;
}
