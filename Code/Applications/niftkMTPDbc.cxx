/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLogImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMedianImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBinariseUsingPaddingImageFilter.h>
#include <itkMultipleDilateImageFilter.h>
#include <itkBinaryUnionWithPaddingImageFilter.h>
#include <itkBinaryIntersectWithPaddingImageFilter.h>
#include <itkPointSet.h>

/*!
 * \file niftkMTPDbc.cxx
 * \page niftkMTPDbc
 * \section niftkMTPDbcSummary Runs multiple-time-point differential bias correction.
 *
 * This program runs multiple-time-point differential bias correction, based on:
 * Emma B. Lewis, Nicholas C. Fox, Correction of differential intensity inhomogeneity in longitudinal MR images, NeuroImage, 
 * Volume 23, Issue 1, September 2004, Pages 75-83, ISSN 1053-8119, 10.1016/j.neuroimage.2004.04.030.
 * 
 * Leung KK, Ridgway GR, Ourselin S, Fox NC, Consistent multi-time-point brain atrophy estimation from the boundary shift integral.
 * Neuroimage. 2012 Feb 15;59(4):3995-4005.
 * 
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float. 
 *
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_INT, "radius", "value", "Radius of the median filter. Default: 5"},
  
  {OPT_INT, "mode", "value", "Determine how the differential bias fields of non-consecutive time-points are calculated. 1: from the images, 2: compose from the differential bias fields of consecutive time-points. Default: 1"},

  {OPT_INT, "expansion", "value", "Expand the bounding box of the union of the masks by the given number of voxels. Default: 5"},

  {OPT_INT, "bspline", "value", "Use bspline to model the differential bias field. Default: 0"},
  
  {OPT_STRING|OPT_LONELY, NULL,  "filename", "Input image 1, input mask 1, output image 1, input image 2, input mask 2, output image 2,..."},
  
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, "Perform the differential bias correction on the two images."}
   
};


enum {
  O_INT_RADIUS=0, 
  
  O_INT_MODE,
  
  O_INT_EXPANSION,

  O_INT_BSPLINE,

  O_FILENAME,
  
  O_MORE

};

/**
 * \brief Differential bias correction. 
 * 
 * Implements "Correction of differential intensity inhomogeneity in longitudinal MR images", Lewis and Fox, 2004, NeuroImage. 
 * 
 * Tries to put the bias of multiple time ponits into the "middle". 
 * 
 */
int main(int argc, char** argv)
{
  int startArg = 1; 
  int inputRadius = 5; 
  int bsplineMode = 0;
  int inputMode = 1; 
  char **multipleStrings = NULL;
  int inputRegionExpansion = 5;
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  CommandLineOptions.GetArgument(O_INT_RADIUS, inputRadius); 
  CommandLineOptions.GetArgument(O_INT_MODE, inputMode); 
  CommandLineOptions.GetArgument(O_INT_EXPANSION, inputRegionExpansion);
  CommandLineOptions.GetArgument(O_INT_BSPLINE, bsplineMode);

  if (inputMode != 1 && inputMode != 2)
  {
    std::cerr << "Mode must be either 1 or 2." << std::endl; 
    return EXIT_FAILURE; 
  }

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, startArg);
  int numberOfStrings = argc - startArg + 1;
  int numberOfImages = numberOfStrings/3; 
  if (numberOfImages <= 1) 
  {
    std::cerr << "Need at least 2 images to perform DBC." << std::endl; 
    return EXIT_FAILURE; 
  }
  std::string* inputFilename = new std::string[numberOfImages]; 
  std::string* inputMaskFilename = new std::string[numberOfImages]; 
  std::string* outputFilename = new std::string[numberOfImages]; 

  std::cout << std::endl << "Input strings: " << std::endl;
  multipleStrings = &argv[startArg-1];
  for (int i=0; i<numberOfStrings; i+=3)
  {
    inputFilename[i/3] = multipleStrings[i]; 
    inputMaskFilename[i/3] = multipleStrings[i+1]; 
    outputFilename[i/3] = multipleStrings[i+2]; 
    std::cout << "   " << i/3 << " " << inputFilename[i/3] << std::endl;
    std::cout << "   " << i/3 << " " << inputMaskFilename[i/3] << std::endl;
    std::cout << "   " << i/3 << " " << outputFilename[i/3] << std::endl;
  }
  
  typedef float PixelType; 
  typedef short MaskPixelType; 
  const int Dimension = 3; 
  typedef itk::Image<PixelType, Dimension>  InputImageType; 
  typedef itk::Image<MaskPixelType, Dimension>  InputMaskType; 
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileReader<InputMaskType> MaskReaderType;
  typedef itk::ImageFileWriter<InputImageType> WriterType;
  typedef itk::ImageRegionIteratorWithIndex<InputImageType> ImageIterator;
  typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ImageConstIterator;
  typedef itk::ImageRegionIteratorWithIndex<InputMaskType> MaskIterator;
  typedef itk::ImageRegionConstIteratorWithIndex<InputMaskType> MaskConstIterator;
  typedef itk::ConstNeighborhoodIterator<InputImageType> ImageConstNeighborhoodIterator;
  typedef itk::NeighborhoodIterator<InputMaskType> MaskNeighborhoodIterator;
  typedef itk::ConstNeighborhoodIterator<InputMaskType> MaskConstNeighborhoodIterator;
  typedef itk::LogImageFilter<InputImageType, InputImageType> LogImageFilterType;
  typedef itk::SubtractImageFilter<InputImageType, InputImageType> SubtractImageFilterType; 
  typedef itk::MedianImageFilter<InputImageType, InputImageType> MedianImageFilterType; 
  typedef itk::ExpImageFilter<InputImageType, InputImageType> ExpImageFilterType; 
  typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyImageFilterType; 
  typedef itk::DivideImageFilter<InputImageType, InputImageType, InputImageType> DivideImageFilterType; 
  typedef itk::ImageDuplicator<InputImageType> DuplicatorType; 
  
  try
  {
    ReaderType::Pointer* inputReader = new ReaderType::Pointer[numberOfImages]; 
    MaskReaderType::Pointer* inputMaskReader = new MaskReaderType::Pointer[numberOfImages]; 
    WriterType::Pointer writer = WriterType::New();
    long double* mean = new long double[numberOfImages];

    // Rectangular region covering the union of all the masks. Initialise it to max
    InputImageType::RegionType requestedRegion;
    for (unsigned int i=0; i<InputImageType::ImageDimension; i++)
    {
      requestedRegion.SetIndex(i, std::numeric_limits<int>::max());
      requestedRegion.SetSize(i, 0);
    }
    std::cout << "requestedRegion=" << requestedRegion << std::endl;

    // Normalisation. Scale the images to the geometric mean of the mean brain intensities of the images. 
    std::cout << "Normalising..." << std::endl; 
    for (int i = 0; i < numberOfImages; i++)
    {
      inputReader[i] = ReaderType::New(); 
      inputReader[i]->SetFileName(inputFilename[i]);
      inputReader[i]->Update(); 
      inputMaskReader[i] = MaskReaderType::New(); 
      inputMaskReader[i]->SetFileName(inputMaskFilename[i]);
      inputMaskReader[i]->Update(); 
      
      long double count1 = 0.0; 
      mean[i] = 0.0; 
      ImageIterator image1It(inputReader[i]->GetOutput(), inputReader[i]->GetOutput()->GetLargestPossibleRegion());
      MaskIterator mask1It(inputMaskReader[i]->GetOutput(), inputMaskReader[i]->GetOutput()->GetLargestPossibleRegion());; 
      for (image1It.GoToBegin(), mask1It.GoToBegin(); 
          !image1It.IsAtEnd(); 
          ++image1It, ++mask1It)
      {
        if (mask1It.Get() > 0)
        {
          mean[i] += image1It.Get(); 
          count1++; 

          for (unsigned int i=0; i<InputImageType::ImageDimension; i++)
          {
            if (requestedRegion.GetIndex(i) > mask1It.GetIndex()[i])
              requestedRegion.SetIndex(i, mask1It.GetIndex()[i]);
            if (requestedRegion.GetIndex(i)+static_cast<int>(requestedRegion.GetSize(i))-1 < mask1It.GetIndex()[i])
              requestedRegion.SetSize(i, mask1It.GetIndex()[i]-requestedRegion.GetIndex(i)+1);
          }
        }
      }
      mean[i] /= count1; 
    }
    std::cout << "requestedRegion=" << requestedRegion << std::endl;
    InputImageType::RegionType largestRegion = inputReader[0]->GetOutput()->GetLargestPossibleRegion();
    for (unsigned int i=0; i<InputImageType::ImageDimension; i++)
    {
      requestedRegion.SetIndex(i, std::max(requestedRegion.GetIndex(i)-inputRegionExpansion, 0L));
      requestedRegion.SetSize(i, std::min(requestedRegion.GetSize(i)+2*inputRegionExpansion, largestRegion.GetSize(i)-requestedRegion.GetIndex(i)));
    }
    std::cout << "expanded requestedRegion=" << requestedRegion << std::endl;

    double normalisedMean = 0.0;
    for (int i = 0; i < numberOfImages; i++)
    { 
      std::cout << "Mean[" << i << "]=" << mean[i] << std::endl;
      if (mean[i] <= 0.0)
      {
        std::cerr << "Mean intensity of input image " << inputFilename[i] << " is -ve:" << mean[i] << std::endl;
        exit(1);
      }
      normalisedMean += vcl_log(mean[i]);
    }
    normalisedMean /= numberOfImages;
    normalisedMean = vcl_exp(normalisedMean);
    std::cout << "normalisedMean=" << normalisedMean << std::endl;

    for (int i = 0; i < numberOfImages; i++)
    {
      ImageIterator image1It(inputReader[i]->GetOutput(), inputReader[i]->GetOutput()->GetLargestPossibleRegion());
      for (image1It.GoToBegin(); !image1It.IsAtEnd(); ++image1It)
      {
        double normalisedValue = normalisedMean*image1It.Get()/mean[i]; 
        if (normalisedValue < 1.0)
          normalisedValue = 1.0; 
        image1It.Set(normalisedValue); 
      }
    }
    
    // Take log. 
    LogImageFilterType::Pointer* logImageFilter = new LogImageFilterType::Pointer[numberOfImages]; 
    
    for (int i = 0; i < numberOfImages; i++)
    {
      std::cout << "Taking log..." << i << std::endl; 
      logImageFilter[i] = LogImageFilterType::New(); 
      logImageFilter[i]->SetInput(inputReader[i]->GetOutput()); 
      logImageFilter[i]->Update(); 
    }
    
    // For all the pairs of images, subtract the log images. 
    SubtractImageFilterType::Pointer* subtractImageFilter = new SubtractImageFilterType::Pointer[numberOfImages*numberOfImages]; 
    for (int i = 0; i < numberOfImages; i++)
    {
      for (int j = i+1; j < numberOfImages; j++)
      {
        std::cout << "Subtracting..." << i << "," << j << std::endl; 
        int index = i*numberOfImages+j; 
        subtractImageFilter[index] = SubtractImageFilterType::New(); 
        subtractImageFilter[index]->SetInput1(logImageFilter[i]->GetOutput()); 
        subtractImageFilter[index]->SetInput2(logImageFilter[j]->GetOutput()); 
        // Just need consecutive pairwise differential bias fields for mode 2. 
        if (inputMode == 2)
          break; 
      }
    }
    
    InputImageType::Pointer* pairwiseBiasRatioImage = new InputImageType::Pointer[numberOfImages*numberOfImages];
    DuplicatorType::Pointer* duplicator = new DuplicatorType::Pointer[numberOfImages*numberOfImages];

    if (bsplineMode == 0)
    {
        // Apply median filter to the subtraction images to obtain the bias.
        MedianImageFilterType::Pointer* medianImageFilter = new MedianImageFilterType::Pointer[numberOfImages*numberOfImages];
        InputImageType::SizeType kernelRadius;
        kernelRadius.Fill(inputRadius);
        for (int i = 0; i < numberOfImages; i++)
        {
          for (int j = i+1; j < numberOfImages; j++)
          {
            std::cout << "Applying median filter..." << i << "," << j << std::endl;
            int index = i*numberOfImages+j;
            medianImageFilter[index] = MedianImageFilterType::New();
            medianImageFilter[index]->SetInput(subtractImageFilter[index]->GetOutput());
            medianImageFilter[index]->SetRadius(kernelRadius);
            medianImageFilter[index]->GetOutput()->SetRequestedRegion(requestedRegion);
            medianImageFilter[index]->Update();
            pairwiseBiasRatioImage[index] = medianImageFilter[index]->GetOutput();

            writer->SetFileName("median_old.img");
            writer->SetInput(medianImageFilter[index]->GetOutput());
            writer->Update();

            // Just need consecutive pairwise differential bias fields for mode 2.
            if (inputMode == 2)
              break;
          }
        }

        // Compose the non-consecutive pairwise differential bias fields form the consecutive ones.
        if (inputMode == 2)
        {
          for (int i = 0; i < numberOfImages; i++)
          {
            for (int j = i+2; j < numberOfImages; j++)
            {
              std::cout << "Composing non-consecutive differential bias field..." << i << "," << j << std::endl;
              int nonConsetiveTimePointIndex = i*numberOfImages+j;
              duplicator[nonConsetiveTimePointIndex] = DuplicatorType::New();
              duplicator[nonConsetiveTimePointIndex]->SetInputImage(inputReader[i]->GetOutput());
              duplicator[nonConsetiveTimePointIndex]->Update();
              pairwiseBiasRatioImage[nonConsetiveTimePointIndex] = duplicator[nonConsetiveTimePointIndex]->GetOutput();
              pairwiseBiasRatioImage[nonConsetiveTimePointIndex]->FillBuffer(0.0);
              for (int m = i, n=i+1; m < j; m++,n++)
              {
                std::cout << "  Adding in..." << m << "," << n << std::endl;
                int consetiveTimePointIndex = m*numberOfImages+n;
                ImageIterator image1It(pairwiseBiasRatioImage[nonConsetiveTimePointIndex], requestedRegion);
                ImageIterator image2It(medianImageFilter[consetiveTimePointIndex]->GetOutput(), requestedRegion);
                for (image1It.GoToBegin(), image2It.GoToBegin();
                    !image1It.IsAtEnd();
                    ++image1It, ++image2It)
                {
                  image1It.Set(image1It.Get()+image2It.Get());
                }
              }
            }
          }
        }
    }
    else
    {

        // The disadvantage of the median filter in the DBC is that the median filter doesn't work too well
        // if the brain is very different between the baseline and repeat scans, i.e. brain atrophy between
        // the two scans will be counted as differential bias.
        //
        // The idea here is to try to use the unchanged region in the brain only. I model the differential
        // bias field in the unchanged region using bspline, and deduce the bias field in changed region
        // from the model.
        //
        // Prepare the region for feeding the points into bspline image filter.
        // 1. Dilate the baseline and repeat brains by n times.
        // 2. Create a union of the dilated regions.
        // 3. Calculate the changed region by subtracting the intersection of the undilated region from
        //    the union of the undilated region.
        // 4. The unchanged region is given by subtracting the changed region from the union of dilated regions.

        DuplicatorType::Pointer* bsplineImage = new DuplicatorType::Pointer[numberOfImages*numberOfImages];
        typedef itk::BinariseUsingPaddingImageFilter<InputMaskType, InputMaskType> BinariseUsingPaddingImageFilterType;
        BinariseUsingPaddingImageFilterType::Pointer binariseUsingPaddingImageFilterI = BinariseUsingPaddingImageFilterType::New();
        BinariseUsingPaddingImageFilterType::Pointer binariseUsingPaddingImageFilterJ = BinariseUsingPaddingImageFilterType::New();
        typedef itk::MultipleDilateImageFilter<InputMaskType> MultipleDilateImageFilterType;
        MultipleDilateImageFilterType::Pointer multipleDilateImageFilterI = MultipleDilateImageFilterType::New();
        MultipleDilateImageFilterType::Pointer multipleDilateImageFilterJ = MultipleDilateImageFilterType::New();
        typedef itk::BinaryUnionWithPaddingImageFilter<InputMaskType, InputMaskType> BinaryUnionWithPaddingImageFilterType;
        BinaryUnionWithPaddingImageFilterType::Pointer binaryUnionWithPaddingDilatedImageFilter = BinaryUnionWithPaddingImageFilterType::New();
        BinaryUnionWithPaddingImageFilterType::Pointer binaryUnionWithPaddingUndilatedImageFilter = BinaryUnionWithPaddingImageFilterType::New();
        typedef itk::BinaryIntersectWithPaddingImageFilter<InputMaskType, InputMaskType> BinaryIntersectWithPaddingImageFilterType;
        BinaryIntersectWithPaddingImageFilterType::Pointer binaryIntersectWithPaddingUndilatedImageFilter = BinaryIntersectWithPaddingImageFilterType::New();
        const unsigned int ParametricDimension = InputImageType::ImageDimension;
        const unsigned int DataDimension = 1;
        typedef itk::Vector<InputImageType::PixelType, DataDimension> VectorType;
        typedef itk::Image<VectorType, ParametricDimension> VectorImageType;
        typedef itk::PointSet<VectorImageType::PixelType, ParametricDimension> PointSetType;
        typedef itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, VectorImageType>  BSplineScatteredDataToImageFilterType;
        const unsigned int maskDilation = 5;

        for (int i = 0; i < numberOfImages; i++)
        {
          binariseUsingPaddingImageFilterI->SetInput(inputMaskReader[i]->GetOutput());
          binariseUsingPaddingImageFilterI->Update();
          multipleDilateImageFilterI->SetInput(binariseUsingPaddingImageFilterI->GetOutput());
          multipleDilateImageFilterI->SetNumberOfDilations(maskDilation);
          multipleDilateImageFilterI->Update();
          for (int j = i+1; j < numberOfImages; j++)
          {
            std::cerr << "Preparing mask..." << i << "," << j << std::endl;
            const int index = i*numberOfImages+j;

            binariseUsingPaddingImageFilterJ->SetInput(inputMaskReader[j]->GetOutput());
            binariseUsingPaddingImageFilterJ->Update();
            multipleDilateImageFilterJ->SetInput(binariseUsingPaddingImageFilterJ->GetOutput());
            multipleDilateImageFilterJ->SetNumberOfDilations(maskDilation);
            multipleDilateImageFilterJ->Update();

            binaryUnionWithPaddingDilatedImageFilter->SetInput(0, multipleDilateImageFilterI->GetOutput());
            binaryUnionWithPaddingDilatedImageFilter->SetInput(1, multipleDilateImageFilterJ->GetOutput());
            binaryUnionWithPaddingDilatedImageFilter->Update();

            binaryUnionWithPaddingUndilatedImageFilter->SetInput(0, binariseUsingPaddingImageFilterI->GetOutput());
            binaryUnionWithPaddingUndilatedImageFilter->SetInput(1, binariseUsingPaddingImageFilterJ->GetOutput());
            binaryUnionWithPaddingUndilatedImageFilter->Update();

            binaryIntersectWithPaddingUndilatedImageFilter->SetInput(0, binariseUsingPaddingImageFilterI->GetOutput());
            binaryIntersectWithPaddingUndilatedImageFilter->SetInput(1, binariseUsingPaddingImageFilterJ->GetOutput());
            binaryIntersectWithPaddingUndilatedImageFilter->Update();

            std::cerr << "Setting up the scatterred data point set..." << std::endl;
            InputImageType::SizeType radius;
            radius.Fill(inputRadius);
            MaskNeighborhoodIterator dilatedUnionImageIterator(radius, binaryUnionWithPaddingDilatedImageFilter->GetOutput(), requestedRegion);
            MaskConstIterator undilatedUnionImageIterator(binaryUnionWithPaddingUndilatedImageFilter->GetOutput(), requestedRegion);
            MaskConstIterator undilatedIntersectImageIterator(binaryIntersectWithPaddingUndilatedImageFilter->GetOutput(), requestedRegion);
            for (;
                 !dilatedUnionImageIterator.IsAtEnd();
                 ++dilatedUnionImageIterator, ++undilatedUnionImageIterator, ++undilatedIntersectImageIterator)
            {
              dilatedUnionImageIterator.SetCenterPixel(dilatedUnionImageIterator.GetCenterPixel() - (undilatedUnionImageIterator.Get() - undilatedIntersectImageIterator.Get()));
            }
            subtractImageFilter[index]->Update();
            ImageConstNeighborhoodIterator subtractionImageIterator(radius, subtractImageFilter[index]->GetOutput(), requestedRegion);
            PointSetType::Pointer pointSet = PointSetType::New();

            DuplicatorType::Pointer filteredImage = DuplicatorType::New();
            filteredImage->SetInputImage(inputReader[i]->GetOutput());
            filteredImage->Update();
            filteredImage->GetOutput()->FillBuffer(0);

            for (dilatedUnionImageIterator.GoToBegin();
                 !dilatedUnionImageIterator.IsAtEnd();
                 ++dilatedUnionImageIterator, ++subtractionImageIterator)
            {
              if (dilatedUnionImageIterator.GetCenterPixel())
              {
                // Median filtering of the subtract image using on the values inside the mask.
                std::vector<PixelType> allValues;
                for (unsigned int n = 0; n < subtractionImageIterator.Size(); ++n)
                {
                    if (dilatedUnionImageIterator.GetPixel(n) > 0)
                        allValues.push_back(subtractionImageIterator.GetPixel(n));
                }
                const unsigned int middle = allValues.size()/2;
                std::nth_element(allValues.begin(), allValues.begin()+middle, allValues.end());
                const PixelType median = allValues[middle];

                PointSetType::PointType point;
                subtractImageFilter[index]->GetOutput()->TransformIndexToPhysicalPoint(subtractionImageIterator.GetIndex(), point);
                const unsigned long count = pointSet->GetNumberOfPoints();
                pointSet->SetPoint(count, point);

                PointSetType::PixelType value(DataDimension);
                value[0] = median; // subtractionImageIterator.Get();
                pointSet->SetPointData(count, value);

                filteredImage->GetOutput()->SetPixel(subtractionImageIterator.GetIndex(), median);
              }
            }

            std::cerr << "Fitting bspline..." << std::endl;
            BSplineScatteredDataToImageFilterType::Pointer bSplineScatteredDataToImageFilter = BSplineScatteredDataToImageFilterType::New();
            // Define the parametric domain
            bSplineScatteredDataToImageFilter->SetOrigin(subtractImageFilter[index]->GetOutput()->GetOrigin());
            bSplineScatteredDataToImageFilter->SetSpacing(subtractImageFilter[index]->GetOutput()->GetSpacing());
            bSplineScatteredDataToImageFilter->SetSize(subtractImageFilter[index]->GetOutput()->GetLargestPossibleRegion().GetSize());
            bSplineScatteredDataToImageFilter->SetInput(pointSet);
            bSplineScatteredDataToImageFilter->SetSplineOrder(3);
            BSplineScatteredDataToImageFilterType::ArrayType ncps;
            ncps.Fill(32);
            bSplineScatteredDataToImageFilter->SetNumberOfControlPoints(ncps);
            bSplineScatteredDataToImageFilter->SetNumberOfLevels(3);
            bSplineScatteredDataToImageFilter->SetGenerateOutputImage(true);
            bSplineScatteredDataToImageFilter->Update();

            bsplineImage[index] = DuplicatorType::New();
            bsplineImage[index]->SetInputImage(inputReader[i]->GetOutput());
            bsplineImage[index]->Update();
            pairwiseBiasRatioImage[index] = bsplineImage[index]->GetOutput();

            itk::ImageRegionConstIteratorWithIndex<VectorImageType> bsplineImageIterator(bSplineScatteredDataToImageFilter->GetOutput(),
                                                                                         bSplineScatteredDataToImageFilter->GetOutput()->GetLargestPossibleRegion());
            ImageIterator biasIterator(pairwiseBiasRatioImage[index], pairwiseBiasRatioImage[index]->GetLargestPossibleRegion());
            for (; !bsplineImageIterator.IsAtEnd(); ++bsplineImageIterator, ++biasIterator)
            {
                biasIterator.Set(bsplineImageIterator.Get()[0]);
            }

            // debug.
            writer->SetFileName("bias.img");
            writer->SetInput(pairwiseBiasRatioImage[index]);
            writer->Update();
            writer->SetFileName("subtract.img");
            writer->SetInput(subtractImageFilter[index]->GetOutput());
            writer->Update();
            writer->SetFileName("median.img");
            writer->SetInput(filteredImage->GetOutput());
            writer->Update();
          }
        }

        DuplicatorType::Pointer* duplicator = new DuplicatorType::Pointer[numberOfImages*numberOfImages];
        // Compose the non-consecutive pairwise differential bias fields form the consecutive ones.
        if (inputMode == 2)
        {
          for (int i = 0; i < numberOfImages; i++)
          {
            for (int j = i+2; j < numberOfImages; j++)
            {
              std::cout << "Composing non-consecutive differential bias field..." << i << "," << j << std::endl;
              int nonConsetiveTimePointIndex = i*numberOfImages+j;
              duplicator[nonConsetiveTimePointIndex] = DuplicatorType::New();
              duplicator[nonConsetiveTimePointIndex]->SetInputImage(inputReader[i]->GetOutput());
              duplicator[nonConsetiveTimePointIndex]->Update();
              pairwiseBiasRatioImage[nonConsetiveTimePointIndex] = duplicator[nonConsetiveTimePointIndex]->GetOutput();
              pairwiseBiasRatioImage[nonConsetiveTimePointIndex]->FillBuffer(0.0);
              for (int m = i, n=i+1; m < j; m++,n++)
              {
                std::cout << "  Adding in..." << m << "," << n << std::endl;
                int consetiveTimePointIndex = m*numberOfImages+n;
                ImageIterator image1It(pairwiseBiasRatioImage[nonConsetiveTimePointIndex], requestedRegion);
                ImageIterator image2It(bsplineImage[consetiveTimePointIndex]->GetOutput(), requestedRegion);
                for (image1It.GoToBegin(), image2It.GoToBegin();
                    !image1It.IsAtEnd();
                    ++image1It, ++image2It)
                {
                  image1It.Set(image1It.Get()+image2It.Get());
                }
              }
            }
          }
        }
    }
    
    // Calculate the bias ratio images in log scale, i.e. R_1 = (r_12*r_13)^(1/3), R_2=c(r_21*r_23)^(1/3), ... etc. 
    DuplicatorType::Pointer* biasRatioImage = new DuplicatorType::Pointer[numberOfImages]; 
    for (int i = 0; i < numberOfImages; i++)
    {
      biasRatioImage[i] = DuplicatorType::New(); 
      biasRatioImage[i]->SetInputImage(inputReader[i]->GetOutput()); 
      biasRatioImage[i]->Update();
      biasRatioImage[i]->GetOutput()->FillBuffer(0.0); 
      for (int j = 0; j < numberOfImages; j++)
      {
        float sign = 1.0; 
        int index = i*numberOfImages+j; 
        if (i > j)   
        {
          sign = -1.0; 
          index = j*numberOfImages+i; 
        }
        if (i != j)
        {
          std::cout << "Calculating bias ratio in log scale..." << i << "," << j << std::endl; 
          ImageIterator image1It(biasRatioImage[i]->GetOutput(), requestedRegion);
          ImageIterator image2It(pairwiseBiasRatioImage[index], requestedRegion);
          for (image1It.GoToBegin(), image2It.GoToBegin(); 
              !image1It.IsAtEnd(); 
              ++image1It, ++image2It)
          {
            image1It.Set(image1It.Get()+(sign*image2It.Get())/numberOfImages); 
          }
        }
      }
    }
    
    
    ExpImageFilterType::Pointer* expImageFilter = new ExpImageFilterType::Pointer[numberOfImages]; 
    DivideImageFilterType::Pointer* divideImageFilter = new DivideImageFilterType::Pointer[numberOfImages]; 
    for (int i = 0; i < numberOfImages; i++)
    {
      // Exponential the the bias ratio. 
      expImageFilter[i] = ExpImageFilterType::New(); 
      expImageFilter[i]->SetInput(biasRatioImage[i]->GetOutput()); 
    
      // Apply the bias to the images by dividing by the bias ratio. 
      divideImageFilter[i] = DivideImageFilterType::New(); 
      divideImageFilter[i]->SetInput1(inputReader[i]->GetOutput()); 
      divideImageFilter[i]->SetInput2(expImageFilter[i]->GetOutput()); 
      
      // Save them. 
      writer->SetFileName(outputFilename[i]);
      writer->SetInput(divideImageFilter[i]->GetOutput());
      writer->Update(); 
    }
        
    delete [] inputReader; 
    delete [] inputMaskReader; 
    delete [] mean;
    delete [] logImageFilter; 
    //delete [] medianImageFilter;
    //delete[] bsplineImage;
    delete [] biasRatioImage; 
    delete [] expImageFilter; 
    delete [] divideImageFilter; 
    delete [] pairwiseBiasRatioImage; 
    delete [] duplicator; 
  }
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Error:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  
  
  
  delete [] inputFilename; 
  delete [] inputMaskFilename; 
  delete [] outputFilename; 
  
  return 0; 
  
}

  
  
  
  
  
