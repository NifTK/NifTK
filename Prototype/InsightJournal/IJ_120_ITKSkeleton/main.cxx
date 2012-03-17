#include <iostream>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>
#include <itkAnalyzeImageIO.h> 
#include <itkJPEGImageIO.h> 
#include <itkPNGImageIO.h> 
#include "itkRescaleIntensityImageFilter.h"

#include "itkSkeletonizeImageFilter.h"
#include "itkChamferDistanceTransformImageFilter.h"

int main(int argc, char** argv)
{
    try
    {
        unsigned int const Dimension = 3;
        typedef itk::Image<unsigned char, Dimension> ImageType;
        itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
        reader->SetFileName("Data/bunnyPadded.hdr");
        reader->Update();
        std::clog << "Image read" << std::endl;

        typedef itk::SkeletonizeImageFilter<ImageType, itk::Connectivity<Dimension, 0> > Skeletonizer;

        typedef itk::ChamferDistanceTransformImageFilter<ImageType, Skeletonizer::OrderingImageType> DistanceMapFilterType;
        DistanceMapFilterType::Pointer distanceMapFilter = DistanceMapFilterType::New();
        unsigned int weights[] = { 3, 4, 5 };
        distanceMapFilter->SetDistanceFromObject(false);
        distanceMapFilter->SetWeights(weights, weights+3);
        distanceMapFilter->SetInput(reader->GetOutput());
        distanceMapFilter->Update();
        std::clog << "Distance map generated" << std::endl;

        Skeletonizer::Pointer skeletonizer = Skeletonizer::New();
        skeletonizer->SetInput(reader->GetOutput());
        skeletonizer->SetOrderingImage(distanceMapFilter->GetOutput());
        skeletonizer->Update();
        ImageType::Pointer skeleton = skeletonizer->GetOutput();
        std::clog << "Skeleton generated" << std::endl;

        itk::ImageFileReader<ImageType>::Pointer reader2 = itk::ImageFileReader<ImageType>::New();
        reader2->SetFileName("Data/bunnySkeleton.hdr");
        reader2->Update();
        ImageType::Pointer reference = reader2->GetOutput();
        if(reference->GetRequestedRegion() != skeleton->GetRequestedRegion())
        {
            std::cerr << "FAILED : image regions are not the same" << "\n";
            std::cerr << skeleton->GetRequestedRegion() << "\n";
            std::cerr << reference->GetRequestedRegion() << "\n";
            return EXIT_FAILURE;
        }
        itk::ImageRegionConstIterator<ImageType> skeletonIt(skeleton, skeleton->GetRequestedRegion());
        itk::ImageRegionConstIterator<ImageType> referenceIt(reference, reference->GetRequestedRegion());
        while(!skeletonIt.IsAtEnd())
        {
            if(skeletonIt.Value() != 0 && referenceIt.Value() ==0 ||
               skeletonIt.Value() == 0 && referenceIt.Value() !=0)
            {
                std::cerr << "FAILED : images are different" << "\n";
                return EXIT_FAILURE;
            }
            ++skeletonIt;
            ++referenceIt;
        }

        std::cout << "PASSED" << "\n";
    }
    catch(std::exception & e)
    {
        std::cerr << "FAILED : " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
