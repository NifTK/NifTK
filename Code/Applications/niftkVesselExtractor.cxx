/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "itkLogHelper.h"


#include <itkCastImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImage.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageDuplicator.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryErodeImageFilter.h>
#include <math.h>

#include <niftkVesselExtractorCLP.h>

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
//    " <name>isTOF</name>\n"
//    " <longflag>tof</longflag>\n"
//    " <description>Input image is TOF.</description>\n"
//    " <label>TOF input</label>\n"
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
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
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
static std::string CLI_PROGRESS_UPDATES = std::string(getenv("NIFTK_CLI_PROGRESS_UPD") != 0 ? getenv("NIFTK_CLI_PROGRESS_UPD") : "");

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
struct arguments
{
    std::string inputImageName;
    std::string outputImageName;
    std::string brainImageName;

    unsigned int scales;
    unsigned int mod;

    float max;
    float min;
    float alphaone;
    float alphatwo;

    bool isBin;
    bool isCT;
    bool isTOF;
    bool doIntensity;

  arguments() {
    scales = 0;
    mod = 0;

    max = 3.09375;
    min = 1;
    alphaone = 0.5;
    alphatwo = 2.0;

    isBin = false;
    isCT = false;
    isTOF = false;
    doIntensity = false;

  }
};

int main( int argc, char *argv[] )
{

    arguments args;


//    // Validate command line args
//    // ~~~~~~~~~~~~~~~~~~~~~~~~~~

    PARSE_ARGS;

    if ( args.inputImageName.length() == 0 || args.outputImageName.length() == 0 )
    {
      commandLine.getOutput()->usage(commandLine);
      return EXIT_FAILURE;
    }

  //Check for the extension
  std::size_t found_nii = args.outputImageName.rfind(".nii");
  std::size_t found_mhd = args.outputImageName.rfind(".mhd");
  if ((found_nii == std::string::npos) && (found_mhd == std::string::npos))
  {
    args.outputImageName += ".nii";
  }

  if (args.mod != 0 && args.mod != 1)
  {
    std::cerr << "Unknown scale mode. Must be 0 (linear) or 1 (exponential)" << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (args.max < 0 || args.min < 0)
  {
    std::cerr << "Maximum/minimum vessel size must be a positive number" << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }


  const unsigned int Dimension = 3;
  typedef short InputPixelType;
  typedef unsigned short OutputPixelType;
  typedef float InternalPixelType;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< InternalPixelType, Dimension > VesselImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  typedef itk::CastImageFilter< VesselImageType, OutputImageType > CastOutFilterType;
  typedef itk::MultiScaleVesselnessFilter< InputImageType, VesselImageType >  VesselnessFilterType;
  typedef itk::IntensityFilter< InputImageType, VesselImageType > IntensityFilterType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType; //TODO Change back
  typedef itk::RescaleIntensityImageFilter< VesselImageType > VesselRescalerType;
  typedef itk::RescaleIntensityImageFilter< InputImageType, VesselImageType >InputRescalerType;
  typedef itk::ResampleImage< InputImageType> InputResampleType;
  typedef itk::ResampleImage< OutputImageType> OutputResampleType;

  startProgress();
  WriterType::Pointer writer = WriterType::New();
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputImageName );
  reader->Update();
  InputImageType::Pointer in_image = reader->GetOutput();
  InputImageType::SpacingType spacing = in_image->GetSpacing();
  InputImageType::SizeType size_in = in_image->GetLargestPossibleRegion().GetSize();

  ReaderType::Pointer mask_reader;
  InputImageType::Pointer mask_image;
  InputImageType::SizeType size_mask;
  bool useMask = false;
  if (args.brainImageName.length() > 0 )
  {
    mask_reader = ReaderType::New();
    mask_reader->SetFileName( brainImageName );
    mask_reader->Update();
    mask_image = mask_reader->GetOutput();
    useMask = true;
    size_mask = mask_image->GetLargestPossibleRegion().GetSize();

    if (size_mask[0] != size_in[0] || size_mask[1] != size_in[1] || size_mask[2] != size_in[2])
    {
      progressXML(0, "Warning: Mask and input image have different dimensions. Ignoring mask...");
      useMask = false;
    }
  }

  if (args.isCT)
    args.min = 0.775438;

  progressXML(0, "Computing scales...");
  float min_spacing = static_cast<float>(spacing[0]);
  float z_spacing = static_cast<float>(spacing[2]);

  if (args.min < min_spacing)
    args.min = min_spacing;

  bool anisotropic = false;
  if ((z_spacing / min_spacing) >=2) // highly anisotropic
    anisotropic = true;

  int tasks = 1; //at least computes the vesselness filter response
  if (useMask || isCT)
    tasks++;
  if (args.isCT && !useMask)
    tasks++;
  if (args.isCT)
    tasks++;
  if (args.isBin)
    tasks++;
  if (args.doIntensity)
    tasks++;
  if (anisotropic)
    tasks+=2;

  tasks++;

  int progress_unit = floor(100.0f / (float)tasks +0.5);
  int progresscounter = progress_unit;

  anisotropic = false;
  if (anisotropic)
  {
    progressXML(progresscounter, "Making image isotropic...");
    progresscounter+=progress_unit;
    double outspacing = (double) (spacing[2] / 2);
    unsigned int outsize = size_in[2] * 2;

    InputResampleType::Pointer in_resample = InputResampleType::New();
    in_resample->SetInput( reader->GetOutput() );
    in_resample->SetAxialSpacing( outspacing );
    in_resample->SetAxialSize( outsize );
    in_resample->Update();
    in_image = in_resample->GetOutput();
    in_image->DisconnectPipeline();

    if (useMask)
    {
      in_resample->SetInput(mask_reader->GetOutput());
      in_resample->Update();
      mask_image = in_resample->GetOutput();
      mask_image->DisconnectPipeline();
    }
  }

  bool neg_img = false;
  if (args.isCT)
  {
    progressXML(progresscounter, "Indentifying if the image is in Hounsfield Units...");
    progresscounter+=progress_unit;

    itk::ImageRegionIterator<InputImageType> inimageIterator(in_image,in_image->GetLargestPossibleRegion());
    while(!inimageIterator.IsAtEnd() && !neg_img) //Check if image comes in HU
    {
      if (inimageIterator.Get() < 0)
        neg_img = true;
      ++inimageIterator;
    }
  }

  //Create a skull mask from CTA
  if (args.isCT && !useMask)
  {
    progressXML(progresscounter, "Creating a skull mask...");
    progresscounter+=progress_unit;

    typedef itk::BrainMaskFromCTFilter<InputImageType,InputImageType> MaskFilterType;
    MaskFilterType::Pointer maskfilter = MaskFilterType::New();
    maskfilter->SetInput(in_image);
    maskfilter->CheckHounsFieldUnitsOff();
    maskfilter->SetIsHU(neg_img);
    maskfilter->Update();
    mask_image = maskfilter->GetOutput();
    useMask = true;

  } //end skull mask

  if (useMask) // Erode for CT, dilate for other modalities!
  {
    typedef itk::BinaryBallStructuringElement<
        InputImageType::PixelType,3>                  StructuringElementType;
    StructuringElementType structuringElement;
    if (args.isCT) {
      structuringElement.SetRadius(3);
      structuringElement.CreateStructuringElement();
      typedef itk::BinaryErodeImageFilter<InputImageType,InputImageType,StructuringElementType> ErodeFilter;
      ErodeFilter::Pointer erode = ErodeFilter::New();
      erode->SetInput( mask_image );
      erode->SetKernel(structuringElement);
      erode->SetErodeValue(1);
       erode->SetBackgroundValue(0);
      erode->Update();
      mask_image = erode->GetOutput();
    }
  }

  progressXML(progresscounter, "Computing vesselness response...");
  progresscounter+=progress_unit;
  VesselnessFilterType::Pointer vesselnessFilter = VesselnessFilterType::New();
  vesselnessFilter->SetInput( in_image );
  vesselnessFilter->SetAlphaOne( args.alphaone );
  vesselnessFilter->SetAlphaTwo( args.alphatwo );
  vesselnessFilter->SetMinScale(args.min );
  vesselnessFilter->SetMaxScale( args.max );
  vesselnessFilter->SetScaleMode(static_cast<VesselnessFilterType::ScaleModeType>(args.mod));
  vesselnessFilter->Update();
  VesselImageType::Pointer maxImage = vesselnessFilter->GetOutput();
  maxImage->DisconnectPipeline();
  itk::ImageRegionIterator<VesselImageType> outimageIterator(maxImage,maxImage->GetLargestPossibleRegion());

  if (useMask && args.isCT)
  {
    progressXML(progresscounter, "Applying mask...");
    progresscounter+=progress_unit;
    itk::ImageRegionConstIterator<InputImageType> maskIterator(mask_image,maxImage->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<InputImageType> inimageIterator(in_image,maxImage->GetLargestPossibleRegion());
    outimageIterator.GoToBegin();
    InputPixelType thresh = 400;
    if (!neg_img)
      thresh = 1324;
    while(!outimageIterator.IsAtEnd()) //Apply brain mask
    {
      if (maskIterator.Get() == 0 || inimageIterator.Get() >= thresh)
        outimageIterator.Set(0);
      ++outimageIterator;
      ++maskIterator;
      ++inimageIterator;
    }
  }
  else if (useMask)
  {
    progressXML(progresscounter, "Applying mask...");
    progresscounter+=progress_unit;
    itk::ImageRegionConstIterator<InputImageType> maskIterator(mask_image,maxImage->GetLargestPossibleRegion());
    outimageIterator.GoToBegin();
    while(!outimageIterator.IsAtEnd()) //Apply brain mask
    {
      if (maskIterator.Get() == 0)
        outimageIterator.Set(0);
      ++outimageIterator;
      ++maskIterator;
    }
  } // end of vesselextractor.cpp

  //parameters
  float min_thresh = 0.003, max_thresh = 1, percentage = 0.04;

  if (args.isTOF)
  {
    min_thresh = 0.003;
    max_thresh = 1;
    percentage = 0.02;
  }
  else
  {
    min_thresh = 0.005;
    max_thresh = 0.5;
    max_thresh = 1;
    min_thresh = 0.03;
    percentage = 0.02;
  }

  if (args.doIntensity)
  {
    progressXML(progresscounter, "Intensity filtering...");
    progresscounter+=progress_unit;
    IntensityFilterType::Pointer intensityfilter = IntensityFilterType::New();
    intensityfilter->SetIntensityImage( in_image );
    intensityfilter->SetVesselnessImage( maxImage );
    intensityfilter->SetFilterMode(static_cast<IntensityFilterType::FilterModeType>(2));
    intensityfilter->Update();
    maxImage->DisconnectPipeline();
    maxImage = intensityfilter->GetOutput();
  }

  if (args.isBin)
  {
    progressXML(progresscounter, "Binarizing image...");
    progresscounter+=progress_unit;

    typedef itk::BinariseVesselResponseFilter<VesselImageType,OutputImageType> BinariseFilter;
    BinariseFilter::Pointer binarise = BinariseFilter::New();
    binarise->SetInput( maxImage );
    binarise->SetLowThreshold(4);
    binarise->Update();
    OutputImageType::Pointer final_image = binarise->GetOutput();

    if (anisotropic)
    {
      progressXML(progresscounter, "Making image anisotropic again...");
      progresscounter+=progress_unit;
      OutputResampleType::Pointer out_resample = OutputResampleType::New();
      out_resample->SetInput( final_image );
      out_resample->SetAxialSpacing( spacing[2] );
      out_resample->SetAxialSize( size_in[2] );
      out_resample->Update();
      writer->SetInput( out_resample->GetOutput() );
    }
    else
      writer->SetInput( final_image ) ;
  }
  else
  {
    progressXML(progresscounter, "Preparing probability map...");
    progresscounter+=progress_unit;
    CastOutFilterType::Pointer caster = CastOutFilterType::New();
    caster->SetInput( maxImage );
    caster->Update();
    if (anisotropic)
    {
      progressXML(progresscounter, "Making image anisotropic again...");
      progresscounter+=progress_unit;
      OutputResampleType::Pointer out_resample = OutputResampleType::New();
      out_resample->SetInput( caster->GetOutput() );
      out_resample->SetAxialSpacing( spacing[2] );
      out_resample->SetAxialSize( size_in[2] );
      out_resample->Update();
      writer->SetInput( out_resample->GetOutput() );
    }
    else
      writer->SetInput( caster->GetOutput() );
  }

  writer->SetFileName( outputImageName );

  try
  {
    writer->Update();
    closeProgress(outputImageName, "Normal exit");
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    closeProgress(outputImageName, "Failed");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
