/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include <niftkLogHelper.h>

#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkCastImageFilter.h>


#include <itkImage.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include <niftkVesselExtractorCLP.h>
#include <itkNifTKImageIOFactory.h>

#include "niftkConversionUtils.h"
#include "itkBrainMaskFromCTFilter.h"
#include "itkIntensityFilter.h"
#include "itkMultiScaleVesselnessFilter.h"
#include "itkBinariseVesselResponseFilter.h"
#include "itkResampleImage.h"


/*!
* \file niftkVesselExtractor.cxx
* \page niftkVesselExtractor
* \section niftkVesselExtractorSummary Applies Sato's vesselness filter to an image using a range of scales.
*/


//std::string xml_vesselextractor=
//    "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
//    "<executable>\n"
//    " <category>Segmentation</category>\n"
//    " <title>niftkVesselExtractor</title>\n"
//    " <description>Vesselness filter</description>\n"
//    " <version>0.0.1</version>\n"
//    " <documentation-url></documentation-url>\n"
//    " <license>BSD</license>\n"
//    " <contributor>Maria A. Zuluaga (UCL)</contributor>\n"
//    " <parameters>\n"
//    " <label>Images</label>\n"
//    " <description>Input and output images</description>\n"
//    " <image fileExtensions=\"nii,nii.gz,mhd\">\n"
//    " <name>inputImageName</name>\n"
//    " <flag>i</flag>\n"
//    " <description>Input image</description>\n"
//    " <label>Input Image</label>\n"
//    " <channel>input</channel>\n"
//    " </image>\n"
//    " <image fileExtensions=\"nii,nii.gz,mhd\">\n"
//    " <name>brainImageName</name>\n"
//    " <flag>b</flag>\n"
//    " <description>Registered brain mask image</description>\n"
//    " <label>Brain Mask Image</label>\n"
//    " <channel>input</channel>\n"
//    " </image>\n"
//    " <image fileExtensions=\"nii,nii.gz,mhd\">\n"
//    " <name>outputImageName</name>\n"
//    " <flag>o</flag>\n"
//    " <description>Output image</description>\n"
//    " <label>Output Image</label>\n"
//    " <default>outputVesselness.nii</default>\n"
//    " <channel>output</channel>\n"
//    " </image>\n"
//    " </parameters>\n"
//    " <parameters>\n"
//    " <label>Scales</label>\n"
//    " <description>Vessel sizes</description>\n"
//    " <float>\n"
//    " <name>max</name>\n"
//    " <longflag>max</longflag>\n"
//    " <description>Maximum vessel size to be detected</description>\n"
//    " <label>Maximum vessel size</label>\n"
//    " <default>3.09375</default>\n"
//    " </float>\n"
//    " <float>\n"
//    " <name>min</name>\n"
//    " <longflag>min</longflag>\n"
//    " <description>Minimum vessel size to be detected</description>\n"
//    " <label>Minimum vessel size</label>\n"
//    " <default>0.775438</default>\n"
//    " </float>\n"
//    " </parameters>\n"
//    " <parameters>\n"
//    " <label>Filter parameters</label>\n"
//    " <description>Vesselness filter configuration parameters</description>\n"
//    " <integer-enumeration>\n"
//    " <name>mode</name>\n"
//    " <longflag>mod</longflag>\n"
//    " <description>Scale generation method: linear (0) or exponential (1)</description>\n"
//    " <label>Mode</label>\n"
//    " <default>0</default>\n"
//    " <element>0</element>\n"
//    " <element>1</element>\n"
//    " </integer-enumeration>\n"
//    " <float>\n"
//    " <name>alphaone</name>\n"
//    " <longflag>aone</longflag>\n"
//    " <description>Alpha 1 parameter from Sato's filter</description>\n"
//    " <label>Alpha one</label>\n"
//    " <default>0.5</default>\n"
//    " </float>\n"
//    " <float>\n"
//    " <name>alphatwo</name>\n"
//    " <longflag>atwo</longflag>\n"
//    " <description>Alpha 2 parameter from Sato's filter</description>\n"
//    " <label>Alpha two</label>\n"
//    " <default>2</default>\n"
//    " </float>\n"
//    " <boolean>\n"
//    " <name>isCT</name>\n"
//    " <longflag>ct</longflag>\n"
//    " <description>Input image is CT.</description>\n"
//    " <label>CT input</label>\n"
//    " <default>false</default>\n"
//    " </boolean>\n"
//    " <boolean>\n"
//    " <name>doIntensity</name>\n"
//    " <longflag>intfil</longflag>\n"
//    " <description>Use image intensity to filter</description>\n"
//    " <label>Intensity filter</label>\n"
//    " <default>true</default>\n"
//    " </boolean>\n"
//    " <boolean>\n"
//    " <name>isBin</name>\n"
//    " <longflag>bin</longflag>\n"
//    " <description>Volume binarisation.</description>\n"
//    " <label>Binarisation</label>\n"
//    " <default>false</default>\n"
//    " </boolean>\n"
//    " </parameters>\n"
//    "</executable>\n";

void Usage(char *exec)
{
  niftk::LogHelper::PrintCommandLineHeader(std::cout);
  std::cout << " " << std::endl;
  std::cout << " Applies Sato's vesselness filter to an image using a range of scales." << std::endl;
  std::cout << " " << std::endl;
  std::cout << " " << exec << " [-i inputFileName -o outputFileName ] [options]" << std::endl;
  std::cout << " " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << " -i <filename> Input image " << std::endl;
  std::cout << " -o <filename> Output image" << std::endl;
  std::cout << "*** [options] ***" << std::endl << std::endl;
  std::cout << " -b <filename> Brain mask " << std::endl;
  std::cout << " --min <float> [0.77] Minimum vessel size"<< std::endl;
  std::cout << " --max <float> [3.09] Maximum vessel size"<< std::endl;
  std::cout << " --mod <int> [0] Linear (0) or exponential (1) scale generation" << std::endl;
  std::cout << " --aone <float> [0.5] Alpha one parameter" << std::endl;
  std::cout << " --atwo <float> [2.0] Alpha two parameter" << std::endl;
  std::cout << " --bin Binarise output" << std::endl;
  std::cout << " --ct Input image is CTA" << std::endl;
  std::cout << " --tof Input image is MR TOF" << std::endl;
  std::cout << " --intfil Extra layer of filtering using image intensities" << std::endl;
}

/* *************************************************************** */
/* *************************************************************** */
static std::string CLI_PROGRESS_UPDATES = 
  std::string(getenv("NIFTK_CLI_PROGRESS_UPD") != 0 ? getenv("NIFTK_CLI_PROGRESS_UPD") : "");

void startProgress()
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    std::cout << "<filter-start>\n";
    std::cout << "<filter-name>niftkVesselExtractor</filter-name>\n";
    std::cout << "<filter-comment>niftkVesselExtractor</filter-comment>\n";
    std::cout << "</filter-start>\n";
    std::cout << std::flush;
  }
}

void progressXML(int p, std::string text)
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    float k = static_cast<float>((float) p / 100);
    std::cout << "<filter-progress>" << k <<"</filter-progress>\n";
    std::cout << std::flush;
  }
}

void closeProgress(std::string img, std::string status)
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    std::cout << "<filter-result name=outputImageName>"  << img << "</filter-result>\n";
    std::cout << "<filter-result name=exitStatusOutput>" << status << "</filter-result>\n";
    std::cout << "<filter-progress>100</filter-progress>\n";
    std::cout << "<filter-end>\n";
    std::cout << "<filter-name>niftkVesselExtractor</filter-name>\n";
    std::cout << "<filter-comment>Finished</filter-comment></filter-end>\n";
    std::cout << std::flush;
  }
}

// some typedefs used throughout the functions
const unsigned int InternalDimension = 3;
typedef float InternalPixelType;
typedef itk::Image<InternalPixelType, InternalDimension> InternalImageType;

// function to cast and crop the input image to the internal type
template<int Dimension, class PixelType>
InternalImageType::Pointer CastAndCropInputImage(std::string fileInputImage, int sliceToKeep)
{
  typedef itk::Image<PixelType, Dimension> InputImageType;   
  typedef itk::ImageFileReader<InputImageType>  InputImageReaderType;
 
  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(fileInputImage);

  try
  {
    std::cout << "Reading input image: " << fileInputImage << std::endl; 
    imageReader->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << std::endl << "ERROR: Failed to read the input image: " << err
	      << std::endl << std::endl; 
    return NULL;
  }      

  typename InternalImageType::Pointer outImage = NULL;

  typename InputImageType::Pointer inImage = imageReader->GetOutput();
  typename InputImageType::RegionType inRegion = inImage->GetLargestPossibleRegion();  
  if (Dimension > InternalDimension)
  {
    typedef itk::ExtractImageFilter<InputImageType, InternalImageType> ExtractImageFilter;
    typename ExtractImageFilter::Pointer extractImage = ExtractImageFilter::New();
    extractImage->SetInput(inImage);
 
    typename InputImageType::RegionType::SizeType regionSize = inRegion.GetSize();
    typename InputImageType::RegionType::IndexType regionIndex;
    regionIndex.Fill(0);

    for (unsigned int i = InternalDimension; i < Dimension; i++)
    {
      regionSize[i] = 0;
      regionIndex[i] = sliceToKeep;
    }

    typename InputImageType::RegionType regionSlice(regionIndex, regionSize);
    extractImage->SetExtractionRegion(regionSlice);
    extractImage->SetDirectionCollapseToSubmatrix();
    extractImage->Update();

    outImage = extractImage->GetOutput();
  }
  else
  {
    typedef itk::CastImageFilter<InputImageType, InternalImageType> CastFilterType;
    typename CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput(inImage);
    castFilter->Update();
    outImage = castFilter->GetOutput();
  }

  outImage->DisconnectPipeline();
  return outImage;
}

template<class InputPixelType, class OutputPixelType>
void CastAndSaveOutputImage(std::string fileOutputImage,  
  typename itk::Image<InputPixelType, InternalDimension>::Pointer outputImage)
{
  typedef itk::Image<OutputPixelType, InternalDimension> OutputImageType;
  typedef itk::ImageFileWriter<OutputImageType>          WriterType; 
  typedef itk::Image<InputPixelType, InternalDimension>  InputImageType;

  typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
  typename CastFilterType::Pointer castFilter = CastFilterType::New();
  castFilter->SetInput(outputImage);
  castFilter->Update();

  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(castFilter->GetOutput());
  writer->SetFileName(fileOutputImage);

  try
  {
    writer->Update();
    closeProgress(fileOutputImage, "Normal exit");
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    closeProgress(fileOutputImage, "Failed");
  }
}

namespace Functor
{
template <class TPixel>
class CreateCTMask
{
  public:
    CreateCTMask(){ m_Thresh = (TPixel) 1; }
    CreateCTMask(TPixel thresh) { m_Thresh = thresh; }
    ~CreateCTMask() {}

    bool operator!=(const CreateCTMask &other) const
    {
      return m_Thresh != other.m_Thresh;
    }
 
    bool operator==(const CreateCTMask & other) const
    {
      return !( *this != other );
    }
 
    inline TPixel operator()(const TPixel & A,
                             const TPixel & B) const
    {
      if (A == 0 || B >= m_Thresh)
        return (TPixel)0;
      else
        return (TPixel)1;
    }

  TPixel m_Thresh;
};
}

  outImage->DisconnectPipeline();
  return outImage;
}

template<class InputPixelType, class OutputPixelType>
void CastAndSaveOutputImage(std::string fileOutputImage,  
  typename itk::Image<InputPixelType, InternalDimension>::Pointer outputImage)
{
  typedef itk::Image<OutputPixelType, InternalDimension> OutputImageType;
  typedef itk::ImageFileWriter<OutputImageType>          WriterType; 
  typedef itk::Image<InputPixelType, InternalDimension>  InputImageType;

  typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
  typename CastFilterType::Pointer castFilter = CastFilterType::New();
  castFilter->SetInput(outputImage);
  castFilter->Update();

  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(castFilter->GetOutput());
  writer->SetFileName(fileOutputImage);

  try
  {
    writer->Update();
    closeProgress(fileOutputImage, "Normal exit");
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    closeProgress(fileOutputImage, "Failed");
  }
}

int main(int argc, char *argv[])
{
  itk::NifTKImageIOFactory::Initialize();
  int extractedSlice = 0;

  PARSE_ARGS;

  if (inputImageName.empty() || outputImageName.empty())
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  //Check for the extension
  std::size_t found_nii = outputImageName.rfind(".nii");
  std::size_t found_mhd = outputImageName.rfind(".mhd");
  if ((found_nii == std::string::npos) && (found_mhd == std::string::npos))
  {
    outputImageName += ".nii";
  }

  if (mode != 0 && mode != 1)
  {
    std::cerr << "Unknown scale mode. Must be 0 (linear) or 1 (exponential)" << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (max < 0 || min < 0)
  {
    std::cerr << "Maximum/minimum vessel size must be a positive number" << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }
  
  startProgress();

  InternalImageType::Pointer inImage = NULL;
  int dims = itk::PeekAtImageDimension(inputImageName);

  if (dims != 3 && dims != 4)
  {
    progressXML(0, "Unsupported image dimension " + std::to_string(dims) + ".");
    return EXIT_FAILURE;
  }


  switch (itk::PeekAtComponentType(inputImageName))
  {
    case itk::ImageIOBase::CHAR:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, char>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, char>(inputImageName, extractedSlice);
      }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, short>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, short>(inputImageName, extractedSlice);
      }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, int>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, int>(inputImageName, extractedSlice);
      }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, long>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, long>(inputImageName, extractedSlice);
      }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, float>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, float>(inputImageName, extractedSlice);
      }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 3)
      {
        inImage = CastAndCropInputImage<3, double>(inputImageName, extractedSlice);
      }
      else if (dims == 4)
      {
        inImage = CastAndCropInputImage<4, double>(inputImageName, extractedSlice);
      }
      break;
    default:
    {
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (inImage.IsNull())
  {
    progressXML(0, "Unsupported image type. Returning...");
    return EXIT_FAILURE;
  }

  InternalImageType::SizeType size_in = inImage->GetLargestPossibleRegion().GetSize();
  InternalImageType::SpacingType spacing = inImage->GetSpacing();

  InternalImageType::Pointer inMask = NULL;
  if (!brainImageName.empty())
  {
    dims = itk::PeekAtImageDimension(brainImageName);
    if (dims != 3 && dims != 4)
    {
      progressXML(0, "Unsupported image dimension " + std::to_string(dims) + ".");
      return EXIT_FAILURE;
    }

  InternalImageType::SizeType size_in = inImage->GetLargestPossibleRegion().GetSize();
  InternalImageType::SpacingType spacing = inImage->GetSpacing();

  InternalImageType::Pointer inMask = NULL;
  if (!brainImageName.empty())
  {
    dims = itk::PeekAtImageDimension(brainImageName);
    if (dims != 3 && dims != 4)
    {
      progressXML(0, "Unsupported image dimension " + std::to_string(dims) + ".");
      return EXIT_FAILURE;
    }

    switch (itk::PeekAtComponentType(brainImageName))
    {
      case itk::ImageIOBase::CHAR:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, char>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, char>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::UCHAR:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, unsigned char>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, unsigned char>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::SHORT:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, short>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, short>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::INT:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, int>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, int>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::LONG:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, long>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, long>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::FLOAT:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, float>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, float>(brainImageName, extractedSlice);
        }
        break;
      case itk::ImageIOBase::DOUBLE:
        if (dims == 3)
        {
          inMask = CastAndCropInputImage<3, double>(brainImageName, extractedSlice);
        }
        else if (dims == 4)
        {
          inMask = CastAndCropInputImage<4, double>(brainImageName, extractedSlice);
        }
        break;
      default:
      {
        std::cerr << "non standard pixel format" << std::endl;
        return EXIT_FAILURE;
      }
    }

    if (inMask.IsNull())
    {
      progressXML(0, "Warning: Unsupported mask image type. Ignoring mask...");
    }

    InternalImageType::SizeType size_mask = inMask->GetLargestPossibleRegion().GetSize();

    if (size_mask[0] != size_in[0] || size_mask[1] != size_in[1] || size_mask[2] != size_in[2])
    {
      progressXML(0, "Warning: Mask and input image have different dimensions. Ignoring mask...");
      inMask = NULL;
    }
  }
  
  progressXML(0, "Computing scales...");

  if (isCT)
  {
    min = 0.775438;
  }

  if (min < spacing[0])
  {
    min = (float) spacing[0];
  }

  bool anisotropic = false;
  if ((spacing[2] / spacing[0]) >= 2) // highly anisotropic
  {
    anisotropic = true;
  }

  int tasks = 1; //at least computes the vesselness filter response
  if (inMask.IsNotNull() || isCT)
  {
    tasks++;
  }
  else if (isCT)
  {
    tasks += 2;
  }

  if (isBin)
  {
    tasks++;
  }
  if (doIntensity)
  {
    tasks++;
  }
  if (anisotropic)
  {
    tasks += 2;
  }

  int progress_unit = floor(100.0f / (float)tasks + 0.5);
  int progresscounter = progress_unit;

  typedef itk::ResampleImage<InternalImageType> InternalResampleType;

  //anisotropic = false;
  if (anisotropic)
  {
    progressXML(progresscounter, "Making image isotropic...");
    progresscounter += progress_unit;
    double outspacing = (double) (spacing[2] / 2);
    unsigned int outsize = size_in[2] * 2;

    InternalResampleType::Pointer in_resample = InternalResampleType::New();
    in_resample->SetInput(inImage);
    in_resample->SetAxialSpacing(outspacing);
    in_resample->SetAxialSize(outsize);
    in_resample->Update();
    inImage = in_resample->GetOutput();

    inImage->DisconnectPipeline();

    if (inMask.IsNotNull())
    {
      in_resample->SetInput(inMask);
      in_resample->Update();
      inMask = in_resample->GetOutput();
      inMask->DisconnectPipeline();
    }
  }

  typedef itk::StatisticsImageFilter<InternalImageType> ImageStatisticsFilter;

  bool negImage = false;
  if (isCT)
  {
    progressXML(progresscounter, "Indentifying if the image is in Hounsfield Units...");
    progresscounter += progress_unit;

    ImageStatisticsFilter::Pointer statisticsFilter = ImageStatisticsFilter::New();
    statisticsFilter->SetInput(inImage);
    statisticsFilter->Update();

    negImage = (statisticsFilter->GetMinimum() < 0);
  }

  typedef itk::Image<unsigned char, InternalDimension> MaskImageType;

  //Create a skull mask from CTA
  if (isCT && inMask.IsNull())
  {
    progressXML(progresscounter, "Creating a skull mask...");
    progresscounter += progress_unit;

    typedef itk::BrainMaskFromCTFilter<InternalImageType, MaskImageType> MaskFilterType;
    MaskFilterType::Pointer maskfilter = MaskFilterType::New();
    maskfilter->SetInput(inImage);
    maskfilter->CheckHounsFieldUnitsOff();
    maskfilter->SetIsHU(negImage);
    maskfilter->Update();

    typedef itk::CastImageFilter<MaskImageType, InternalImageType> CastFilterType;
    CastFilterType::Pointer castFilter = CastFilterType::New();

    castFilter->SetInput(maskfilter->GetOutput());
    castFilter->Update();
    inMask = castFilter->GetOutput();
  }

  // Erode for CT, dilate for other modalities!
  if (inMask.IsNotNull()) 
  {
    typedef itk::BinaryBallStructuringElement<
      InternalImageType::PixelType, 
      InternalDimension> StructuringElementType;

    StructuringElementType structuringElement;
    structuringElement.SetRadius(3);
    structuringElement.CreateStructuringElement();

    typedef itk::BinaryMorphologyImageFilter<
      InternalImageType, 
      InternalImageType,
      StructuringElementType> MorphFilterType;

    MorphFilterType::Pointer morphFilter;
    if (isCT) 
    {
      typedef itk::BinaryErodeImageFilter<
        InternalImageType, 
        InternalImageType, 
        StructuringElementType> ErodeFilter;
      morphFilter = ErodeFilter::New();
    }
    else
    {
      typedef itk::BinaryDilateImageFilter<
        InternalImageType, 
        InternalImageType, 
        StructuringElementType> DilateFilter;
      morphFilter = DilateFilter::New();
    }

    morphFilter->SetInput(inMask);
    morphFilter->SetKernel(structuringElement);
    morphFilter->SetBackgroundValue(0);
    morphFilter->SetForegroundValue(1);
    morphFilter->Update();
    inMask = morphFilter->GetOutput();
  }

  progressXML(progresscounter, "Computing vesselness response...");
  progresscounter += progress_unit;

  typedef itk::MultiScaleVesselnessFilter<
    InternalImageType, 
    InternalImageType> VesselnessFilterType;
  VesselnessFilterType::Pointer vesselnessFilter = VesselnessFilterType::New();

  vesselnessFilter->SetInput(inImage);
  vesselnessFilter->SetAlphaOne(alphaone);
  vesselnessFilter->SetAlphaTwo(alphatwo);
  vesselnessFilter->SetMinScale(min);
  vesselnessFilter->SetMaxScale(max);
  vesselnessFilter->SetScaleMode(static_cast<VesselnessFilterType::ScaleModeType>(mode));
  vesselnessFilter->Update();

  InternalImageType::Pointer maxImage = vesselnessFilter->GetOutput();
  maxImage->DisconnectPipeline();

  if (inMask.IsNotNull())
  {
    if (isCT)
    {
      progressXML(progresscounter, "Applying mask...");
      progresscounter += progress_unit;

      typedef Functor::CreateCTMask<InternalPixelType> CreateCTMaskFunctor;
      typedef itk::BinaryFunctorImageFilter<
        InternalImageType, 
        InternalImageType, 
        InternalImageType,
        CreateCTMaskFunctor> CreateMaskFilterType;

      CreateMaskFilterType::Pointer createMaskFilter = CreateMaskFilterType::New();

      InternalPixelType thresh = 400;
      if (!negImage)
      {
        thresh = 1324;
      }
      CreateCTMaskFunctor createMask(thresh);
      createMaskFilter->SetFunctor(createMask);

      createMaskFilter->SetInput1(inMask);
      createMaskFilter->SetInput2(inImage);
      createMaskFilter->Update();

      inMask = createMaskFilter->GetOutput();
      inMask->DisconnectPipeline();
    }

    progressXML(progresscounter, "Applying mask...");
    progresscounter += progress_unit;

    typedef itk::MultiplyImageFilter<InternalImageType> MultiplyFilterType;
    MultiplyFilterType::Pointer multiplyFilter = MultiplyFilterType::New();
    multiplyFilter->SetInput1(maxImage);
    multiplyFilter->SetInput2(inMask);
    multiplyFilter->Update();

    maxImage = multiplyFilter->GetOutput();
  } // end of vesselextractor.cpp

  if (doIntensity)
  {
    ImageStatisticsFilter::Pointer statisticsFilter = ImageStatisticsFilter::New();
    statisticsFilter->SetInput(maxImage);
    statisticsFilter->Update();

    progressXML(progresscounter, "Intensity filtering...");
    progresscounter += progress_unit;

    typedef itk::IntensityFilter<InternalImageType, InternalImageType> IntensityFilterType;
    IntensityFilterType::Pointer intensityfilter = IntensityFilterType::New();
    intensityfilter->SetIntensityImage(inImage);
    intensityfilter->SetVesselnessImage(maxImage);
    intensityfilter->SetFilterMode(static_cast<IntensityFilterType::FilterModeType>(2));
    intensityfilter->SetOutputMaximum(statisticsFilter->GetMaximum());
    intensityfilter->Update();

    maxImage->DisconnectPipeline();
    maxImage = intensityfilter->GetOutput();
  }

  if (isBin)
  {
    progressXML(progresscounter, "Binarizing image...");
    progresscounter += progress_unit;

    typedef itk::BinariseVesselResponseFilter<InternalImageType, MaskImageType> BinariseFilter;
    BinariseFilter::Pointer binarise = BinariseFilter::New();
    
    binarise->SetInput(maxImage);
    binarise->SetLowThreshold(4);
    binarise->Update();
    MaskImageType::Pointer maskImage = binarise->GetOutput();

    if (anisotropic)
    {
      progressXML(progresscounter, "Making image anisotropic again...");
      progresscounter+=progress_unit;
      
      typedef itk::ResampleImage<MaskImageType> MaskResampleType;
      MaskResampleType::Pointer out_resample = MaskResampleType::New();

      out_resample->SetInput(maskImage);
      out_resample->SetAxialSpacing(spacing[2]);
      out_resample->SetAxialSize(size_in[2]);

      out_resample->Update();
      maskImage = out_resample->GetOutput();
    }

    CastAndSaveOutputImage<unsigned char, unsigned char>(outputImageName, maskImage);
    return EXIT_SUCCESS;
  }
  else
  {
    progressXML(progresscounter, "Preparing probability map...");
    progresscounter += progress_unit;

    if (anisotropic)
    {
      progressXML(progresscounter, "Making image anisotropic again...");
      progresscounter += progress_unit;
      InternalResampleType::Pointer out_resample = InternalResampleType::New();
      out_resample->SetInput(maxImage);
      out_resample->SetAxialSpacing(spacing[2]);
      out_resample->SetAxialSize(size_in[2]);
      out_resample->Update();
      maxImage = out_resample->GetOutput();
    }

    // use the same type as the input image
    switch (itk::PeekAtComponentType(inputImageName))
    {
      case itk::ImageIOBase::CHAR:
        CastAndSaveOutputImage<InternalPixelType, char>(outputImageName, maxImage);
        break;
      case itk::ImageIOBase::SHORT:
        CastAndSaveOutputImage<InternalPixelType, short>(outputImageName, maxImage);
        break;
      case itk::ImageIOBase::INT:
        CastAndSaveOutputImage<InternalPixelType, int>(outputImageName, maxImage);
        break;
      case itk::ImageIOBase::LONG:
        CastAndSaveOutputImage<InternalPixelType, long>(outputImageName, maxImage);
        break;
      case itk::ImageIOBase::FLOAT:
        CastAndSaveOutputImage<InternalPixelType, float>(outputImageName, maxImage);
        break;
      case itk::ImageIOBase::DOUBLE:
        CastAndSaveOutputImage<InternalPixelType, double>(outputImageName, maxImage);
        break;
    }
  }
  return EXIT_SUCCESS;
}
