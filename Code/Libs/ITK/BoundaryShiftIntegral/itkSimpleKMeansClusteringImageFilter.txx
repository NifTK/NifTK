/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_TXX_
#define ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_TXX_

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkBinariseUsingPaddingImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultipleErodeImageFilter.h>

namespace itk 
{


template <class TInputImage, class TInputMask, class TOutputImage>
void
SimpleKMeansClusteringImageFilter<TInputImage, TInputMask, TOutputImage>
::EstimateIntensityFromDilatedMask(double& csfMean, double& csfSd, double& gmMean, double& wmMean)
{
    typename TInputImage::ConstPointer inputImage = this->GetInput();

    typedef BinariseUsingPaddingImageFilter<TInputMask,TInputMask> BinariseUsingPaddingType;
    typename BinariseUsingPaddingType::Pointer binariseUsingPadding = BinariseUsingPaddingType::New();
    binariseUsingPadding->SetInput(this->m_InputMask);
    binariseUsingPadding->SetPaddingValue(0);
    binariseUsingPadding->Update();

    typedef itk::MultipleDilateImageFilter<TInputMask> MultipleDilateImageFilterType;
    typename MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();

    // Dilate multiple times.
    multipleDilateImageFilter->SetNumberOfDilations(3);
    multipleDilateImageFilter->SetInput(binariseUsingPadding->GetOutput());
    multipleDilateImageFilter->Update();

    // Rough estimate for CSF mean by taking the mean values of the (dilated region - brain region).
    typedef SubtractImageFilter<TInputMask, TInputMask> SubtractImageFilterType;
    typename SubtractImageFilterType::Pointer subtractImageFilter = SubtractImageFilterType::New();
    subtractImageFilter->SetInput1(multipleDilateImageFilter->GetOutput());
    subtractImageFilter->SetInput2(binariseUsingPadding->GetOutput());
    subtractImageFilter->Update();

    typedef itk::MultipleErodeImageFilter<TInputMask> MultipleErodeImageFilterType;
    typename MultipleErodeImageFilterType::Pointer multipleErodeImageFilterFilter = MultipleErodeImageFilterType::New();

    // Erode multiple times.
    multipleErodeImageFilterFilter->SetNumberOfErosions(3);
    multipleErodeImageFilterFilter->SetInput(binariseUsingPadding->GetOutput());
    multipleErodeImageFilterFilter->Update();

    // Rough estimate for GM mean by taking the mean values of the (brain region - eroded region).
    typename SubtractImageFilterType::Pointer gmSubtractImageFilter = SubtractImageFilterType::New();
    gmSubtractImageFilter->SetInput1(binariseUsingPadding->GetOutput());
    gmSubtractImageFilter->SetInput2(multipleErodeImageFilterFilter->GetOutput());
    gmSubtractImageFilter->Update();

    ImageRegionConstIterator<TInputImage> imageIterator(inputImage, inputImage->GetLargestPossibleRegion());
    ImageRegionConstIterator<TInputMask> maskIterator(subtractImageFilter->GetOutput(), subtractImageFilter->GetOutput()->GetLargestPossibleRegion());
    ImageRegionConstIterator<TInputMask> gmMaskIterator(gmSubtractImageFilter->GetOutput(), gmSubtractImageFilter->GetOutput()->GetLargestPossibleRegion());
    ImageRegionConstIterator<TInputMask> wmMaskIterator(multipleErodeImageFilterFilter->GetOutput(), multipleErodeImageFilterFilter->GetOutput()->GetLargestPossibleRegion());

    csfMean = 0.0;
    int count = 0;
    gmMean = 0.0;
    int gmCount = 0;
    wmMean = 0.0;
    int wmCount = 0;

    for (imageIterator.GoToBegin(), maskIterator.GoToBegin(), gmMaskIterator.GoToBegin(), wmMaskIterator.GoToBegin();
         !imageIterator.IsAtEnd();
         ++imageIterator, ++maskIterator, ++gmMaskIterator, ++wmMaskIterator)
    {
        if (maskIterator.Get() > 0)
        {
            csfMean += imageIterator.Get();
            count++;
        }
        if (gmMaskIterator.Get() > 0)
        {
            gmMean += imageIterator.Get();
            gmCount++;
        }
        if (wmMaskIterator.Get() > 0)
        {
            wmMean += imageIterator.Get();
            wmCount++;
        }
    }
    csfMean /= count;
    gmMean /= gmCount;
    wmMean /= wmCount;
    csfSd = 0.0;
    for (imageIterator.GoToBegin(), maskIterator.GoToBegin();
         !imageIterator.IsAtEnd();
         ++imageIterator, ++maskIterator)
    {
        if (maskIterator.Get() > 0)
        {
            double diff = imageIterator.Get()-csfMean;
            csfSd += diff*diff;
        }

    }
    csfSd = sqrt(csfSd/count);

    // std::cerr << "csfMean=" << csfMean << ", gmMean=" << gmMean << ", wmMean=" << wmMean << std::endl;

}


template <class TInputImage, class TInputMask, class TOutputImage>
void
SimpleKMeansClusteringImageFilter<TInputImage, TInputMask, TOutputImage>
::GenerateData()
{
    if (this->m_IsEstimateInitValuesUsingMask)
    {
        double dummy;
        if (this->m_NumberOfClasses == 3)
        {
            EstimateIntensityFromDilatedMask(m_InitialMeans[0], dummy, m_InitialMeans[1], m_InitialMeans[2]);
        }
        else if (this->m_NumberOfClasses == 2)
        {
            EstimateIntensityFromDilatedMask(m_InitialMeans[0], dummy, m_InitialMeans[1], dummy);
        }
    }

    typename TInputImage::ConstPointer inputImage = this->GetInput();
    itk::ImageRegionConstIterator<TInputImage> inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<TInputMask> inputMaskIterator(this->m_InputMask, this->m_InputMask->GetLargestPossibleRegion());
    SampleType::Pointer inputSample = SampleType::New() ;

    // Build up the sample.
    inputImageIterator.GoToBegin();
    inputMaskIterator.GoToBegin();
    for ( ; !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++inputMaskIterator )
    {
        if (inputMaskIterator.Get() > 0)
        {
            typename TInputImage::PixelType value = inputImageIterator.Get();
            MeasurementVectorType oneSample;

            // TODO: quick hack - remove the -ve number from ITK transformed images.
            if (value < 0)
                value = 0;
            oneSample[0] = static_cast<double>(value);
            inputSample->PushBack(oneSample);
        }
    }

    TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();

    // Prepare the K-d tree.
    treeGenerator->SetSample(inputSample);
    treeGenerator->SetBucketSize(16);
    treeGenerator->Update();

    typename EstimatorType::Pointer estimator = EstimatorType::New();

    // K-Means clustering.
    estimator->SetParameters(this->m_InitialMeans);
    estimator->SetKdTree(treeGenerator->GetOutput());
    estimator->SetMaximumIteration(500);
    estimator->SetCentroidPositionChangesThreshold(0.0);
    estimator->StartOptimization();

    this->m_FinalMeans = estimator->GetParameters();
    this->m_FinalStds.SetSize(this->m_NumberOfClasses);
    this->m_FinalClassSizes.SetSize(this->m_NumberOfClasses);
    for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++)
    {
        this->m_FinalStds[classIndex] = 0.0;
        this->m_FinalClassSizes[classIndex] = 0.0;
    }

    // Allocate the output image.
    typename TOutputImage::Pointer outputImage = this->GetOutput();

    outputImage->SetRequestedRegion(this->GetInput()->GetLargestPossibleRegion());
    this->AllocateOutputs();

    itk::ImageRegionIterator<TOutputImage> outputImageIterator(outputImage, outputImage->GetLargestPossibleRegion());

    // Classify each voxel according the distance to the means.
    inputImageIterator.GoToBegin();
    inputMaskIterator.GoToBegin();
    outputImageIterator.GoToBegin();
    this->m_RSS = 0.0;
    this->m_NumberOfSamples = 0.0;
    for ( ; !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++inputMaskIterator, ++outputImageIterator )
    {
        if (inputMaskIterator.Get() > 0)
        {
            typename TInputImage::PixelType value = inputImageIterator.Get();
            int bestClass = -1;
            double bestDistance = std::numeric_limits<double>::max();

            // TODO: quick hack - remove the -ve number from ITK transformed images.
            if (value < 0)
                value = 0;
            for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++ )
            {
                double currentDistance = fabs(value-this->m_FinalMeans[classIndex]);

                if ( currentDistance < bestDistance )
                {
                    bestDistance = currentDistance;
                    bestClass = classIndex;
                }
            }
            this->m_RSS += bestDistance*bestDistance;
            this->m_NumberOfSamples++;
            this->m_FinalStds[bestClass] = this->m_FinalStds[bestClass]+(this->m_FinalMeans[bestClass]-value)*(this->m_FinalMeans[bestClass]-value);
            this->m_FinalClassSizes[bestClass] = this->m_FinalClassSizes[bestClass]+1;
            outputImageIterator.Set(bestClass+1);
        }
        else
        {
            outputImageIterator.Set(0);
        }
    }

    for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++ )
    {
        this->m_FinalStds[classIndex] = sqrt(this->m_FinalStds[classIndex]/this->m_FinalClassSizes[classIndex]);
        //std::cout << this->m_FinalMeans[classIndex] << std::endl << this->m_FinalStds[classIndex] << std::endl;
    }
}


}

#endif
