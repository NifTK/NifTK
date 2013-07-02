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
#include <itkHessian3DToVesselnessMeasureImageFilter.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkSymmetricSecondRankTensor.h>
#include <itkJoinSeriesImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImage.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

/*!
 * \file niftkVesselExtractor.cxx
 * \page niftkVesselExtractor
 * \section niftkVesselExtractorSummary Applies Sato's vesselness filter to an image using a range of scales.
 */


std::string xml_vesselextractor=
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<executable>\n"
"   <category>Segmentation</category>\n"
"   <title>niftkVesselExtractor</title>\n"
"   <description>Vesselness filter</description>\n"
"   <version>0.0.1</version>\n"
"   <documentation-url></documentation-url>\n"
"   <license>BSD</license>\n"
"   <contributor>Maria A. Zuluaga (UCL)</contributor>\n"
"   <parameters>\n"
"      <label>Images</label>\n"
"      <description>Input and output images</description>\n"
"      <image fileExtensions=\"*.nii,*.nii.gz,*.mhd\">\n"
"          <name>inputImageName</name>\n"
"          <flag>i</flag>\n"
"          <description>Input image</description>\n"
"          <label>Input Image</label>\n"
"          <channel>input</channel>\n"
"      </image>\n"
"      <image fileExtensions=\"*.nii,*.nii.gz,*.mhd\">\n"
"          <name>brainImageName</name>\n"
"          <flag>b</flag>\n"
"          <description>Registered brain mask image</description>\n"
"          <label>Brain Mask Image</label>\n"
"          <channel>input</channel>\n"
"      </image>\n"
"     <image fileExtensions=\"*.nii,*.nii.gz,*.mhd\">\n"
"          <name>outputImageName</name>\n"
"          <flag>o</flag>\n"
"          <description>Output image</description>\n"
"          <label>Output Image</label>\n"
"          <default>outputVesselness.nii</default>\n"
"          <channel>output</channel>\n"
"      </image>\n"
"   </parameters>\n"
"   <parameters>\n"
"      <label>Scales</label>\n"
"      <description>Scales to evaluate</description>\n"
"     <integer>\n"
"         <name>scales</name>\n"
"         <longflag>ns</longflag>\n"
"         <description>Number of scales in which the filter will be evaluated</description>\n"
"         <label>Number of scales</label>\n"
"         <default>8</default>\n"
"     </integer>\n"
"   </parameters>\n"
"   <parameters>\n"
"      <label>Filter parameters</label>\n"
"      <description>Vesselness filter configuration parameters</description>\n"
"     <integer-enumeration>\n"
"         <name>mode</name>\n"
"         <longflag>mod</longflag>\n"
"         <description>Scale generation method: linear (0) or exponential (1)</description>\n"
"         <label>Mode</label>\n"
"         <default>0</default>\n"
"	  <element>0</element>\n"
"         <element>1</element>\n"
"     </integer-enumeration>\n"
"      <float>\n"
"         <name>alphaone</name>\n"
"         <longflag>aone</longflag>\n"
"         <description>Alpha 1 parameter from Sato's filter</description>\n"
"         <label>Alpha one</label>\n"
"         <default>0.5</default>\n"
"      </float>\n"
"      <float>\n"
"         <name>alphatwo</name>\n"
"         <longflag>atwo</longflag>\n"
"         <description>Alpha 2 parameter from Sato's filter</description>\n"
"         <label>Alpha two</label>\n"
"         <default>2</default>\n"
"      </float>\n"
"      <float>\n"
"         <name>max</name>\n"
"         <longflag>max</longflag>\n"
"         <description>maximum desired scale when using exponential generator</description>\n"
"         <label>Maximum Scale</label>\n"
"         <default>8</default>\n"
"      </float>\n"
"    <boolean>\n"
"        <name>isBin</name>\n"
"        <longflag>bin</longflag>\n"
"        <description>Volume binarisation.</description>\n"
"        <label>Binarisation</label>\n"
"        <default>1</default>\n"
"    </boolean>\n"
"    <boolean>\n"
"        <name>isCT</name>\n"
"        <longflag>ct</longflag>\n"
"        <description>Input image is CT.</description>\n"
"        <label>CT input</label>\n"
"        <default>1</default>\n"
"    </boolean>\n"
"   </parameters>\n"
"</executable>\n";

void Usage(char *exec)
{
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Applies Sato's vesselness filter to an image using a range of scales." << std::endl;
    std::cout << "  " << std::endl;
    std::cout  << "  " << exec << " [-i inputFileName -b brainmaskFileName -o outputFileName -ns NumberofScales ] [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input  image " << std::endl;
    std::cout << "    -b    <filename>        Brain mask " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl;
    std::cout << "    --ns    <int>        Number of scales" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    --mod    <int>   [0]     Linear (0) or exponential (1) scale generation" << std::endl;
    std::cout << "    --aone   <float> [0.5]   Alpha one parameter" << std::endl;
    std::cout << "    --atwo   <float> [2.0]   Alpha two parameter" << std::endl;
    std::cout << "    --max    <float> [8]    Maximum scale for exponential mode" << std::endl;
    std::cout << "    --bin   Binarise output" << std::endl;
    std::cout << "    --ct   Input image is CTA" << std::endl;
}
  

int main( int argc, char *argv[] )
{
    // Define command line params
    std::string inputImageName;
    std::string outputImageName;
    std::string brainImageName;
    unsigned int scales = 0;
    unsigned int mod = 0;
    float max = 10;
    float alphaone = 0.5;
    float alphatwo = 2.0; 
    bool isBin = false;
    bool isCT = false;
    
    // Parse command line args
    for(int i=1; i < argc; i++){
	if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
	    Usage(argv[0]);
	    return -1;
	}
	else if(strcmp(argv[i], "-i") == 0){
	    inputImageName=argv[++i];
	    std::cout << "Set -i=" << inputImageName << std::endl;
	}
	else if(strcmp(argv[i], "-o") == 0){
	    outputImageName=argv[++i];
	    std::cout << "Set -o=" << outputImageName << std::endl;
	}
    else if(strcmp(argv[i], "-b") == 0){
	    brainImageName=argv[++i];
	    std::cout << "Set -b=" << brainImageName << std::endl;
	}
	else if(strcmp(argv[i], "--ns") == 0){
	    scales=atoi(argv[++i]);
	    std::cout << "Set -ns=" << niftk::ConvertToString(scales) << std::endl;
	}
	else if(strcmp(argv[i], "--mod") == 0){
	    mod=atoi(argv[++i]);
	    std::cout << "Set -mod=" << niftk::ConvertToString(mod) << std::endl;
	}
	else if(strcmp(argv[i], "--aone") == 0){
	    alphaone=atof(argv[++i]);
	    std::cout << "Set -aone=" << niftk::ConvertToString(alphaone) << std::endl;
	}
	else if(strcmp(argv[i], "--atwo") == 0){
	    alphatwo=atof(argv[++i]);
	    std::cout << "Set -atwo=" << niftk::ConvertToString(alphatwo) << std::endl;
	}
	else if(strcmp(argv[i], "--max") == 0){
	    max=atof(argv[++i]);
	    std::cout << "Set -max=" << niftk::ConvertToString(max) << std::endl;
	}
    /*else if(strcmp(argv[i], "--ct") == 0){
	    isCT = true;
	    std::cout << "Set -ct=ON " << std::endl;
	}*/
    else if(strcmp(argv[i], "--bin") == 0){
	    isBin=true;
	    std::cout << "Set -bin=ON" << std::endl;
	}
    else if(strcmp(argv[i], "--ct") == 0){
        isCT=true;
        std::cout << "Set -ct=ON" << std::endl;
    }
	else if(strcmp(argv[i], "--xml") == 0){
	    std::cout << xml_vesselextractor;
	    return EXIT_SUCCESS;
	}
	else {
	    std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
	    return -1;
	}
    }
 
    // Validate command line args
    if (inputImageName.length() == 0 || outputImageName.length() == 0)
    {
	Usage(argv[0]);
	return EXIT_FAILURE;
    }
    
    if (mod != 0 && mod != 1)
    {
	std::cerr << "Unknown scale mode. Must be 0 (linear) or 1 (exponential)" << std::endl;
	Usage(argv[0]);
	return EXIT_FAILURE;
    }
    
    if (scales == 0)
    {
	std::cerr << "Number of scales is mandatory" << std::endl;
	Usage(argv[0]);
	return EXIT_FAILURE;
    }
    
    if (max < 0)
    {
	std::cerr << "Maximum scale must be a positive number" << std::endl;
	Usage(argv[0]);
	return EXIT_FAILURE;
    }
    
    
    const unsigned int Dimension = 3;
    typedef   short    InputPixelType;
    typedef   unsigned short    OutputPixelType;
    typedef   float   InternalPixelType;
  
    typedef   itk::Image< InputPixelType, Dimension >   InputImageType;
    typedef   itk::Image< InternalPixelType, Dimension >   VesselImageType;
    typedef   itk::Image< OutputPixelType, Dimension >  OutputImageType;

    
    typedef   itk::CastImageFilter< InputImageType, VesselImageType > CastFilterType;
    typedef   itk::CastImageFilter< VesselImageType, OutputImageType > CastOutFilterType;
    typedef   itk::HessianRecursiveGaussianImageFilter< 
                            VesselImageType >              HessianFilterType;
    typedef   itk::Hessian3DToVesselnessMeasureImageFilter<
              InternalPixelType > VesselnessMeasureFilterType;
    typedef   itk::ImageFileReader< InputImageType >  ReaderType;
    typedef   itk::ImageFileWriter< OutputImageType > WriterType;
    typedef itk::RescaleIntensityImageFilter< VesselImageType > VesselRescalerType;
    typedef itk::RescaleIntensityImageFilter< InputImageType, VesselImageType >InputRescalerType;
    
   
    WriterType::Pointer   writer = WriterType::New();
    ReaderType::Pointer   reader = ReaderType::New();			    			    
    reader->SetFileName( inputImageName );
    reader->Update();
    InputImageType::Pointer in_image = reader->GetOutput();

    ReaderType::Pointer   mask_reader;
    InputImageType::Pointer mask_image;
    bool useMask = false;
    if (brainImageName.length() > 0 )
    {
        mask_reader = ReaderType::New();
        mask_reader->SetFileName( brainImageName );
        mask_reader->Update();
        mask_image = mask_reader->GetOutput();
        useMask = true;
    }


     
    InputImageType::SpacingType spacing = in_image->GetSpacing();
    InputImageType::SizeType size_in = in_image->GetLargestPossibleRegion().GetSize();
         
    float min_spacing = static_cast<float>(spacing[0]);
      
     std::vector<float> all_scales(scales,0);
     switch (mod)
     {
	 case 0: 
	 {
         for (unsigned int s = 1; s < scales; ++s)
	     {
		all_scales[s] = static_cast<float>(s+1) * min_spacing;
	     }
	     break;
	 }
	 case 1:
	 {
	     float factor = log(max / min_spacing) / static_cast<float>(scales -1);
	     all_scales[0] = min_spacing;
	     for (unsigned int s = 1; s < scales; ++s) 
	     {
		 all_scales[s] = min_spacing * exp(factor * s);
	     }
	     break;
	 }
	 default:
	 {
	     std::cerr << "Error: Unknown mode option" << std::endl;
	     exit(-1);
	 }
     }
    
    CastFilterType::Pointer caster = CastFilterType::New();
    HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
    VesselnessMeasureFilterType::Pointer vesselnessFilter = 
                            VesselnessMeasureFilterType::New();
    VesselImageType::Pointer vesselnessImage;


    caster->SetInput( in_image );
    hessianFilter->SetInput( caster->GetOutput() );
    hessianFilter->SetNormalizeAcrossScale( true );
    vesselnessFilter->SetInput( hessianFilter->GetOutput() );
    vesselnessFilter->SetAlpha1( static_cast< double >(alphaone) );
    vesselnessFilter->SetAlpha2( static_cast< double >(alphatwo) );
    hessianFilter->SetSigma( static_cast< double >( all_scales[0] ) );
    vesselnessFilter->Update();

    VesselImageType::Pointer maxImage = vesselnessFilter->GetOutput();
    maxImage->DisconnectPipeline();

    itk::ImageRegionIterator<VesselImageType> outimageIterator(maxImage,maxImage->GetLargestPossibleRegion());

    for (size_t s = 1; s < all_scales.size(); ++s)
    {
        hessianFilter->SetSigma( static_cast< double >( all_scales[s] ) );
        vesselnessFilter->Update();
        vesselnessImage = vesselnessFilter->GetOutput();

        itk::ImageRegionConstIterator<VesselImageType> vesselimageIterator(vesselnessImage,maxImage->GetLargestPossibleRegion());

        vesselimageIterator.GoToBegin();
        outimageIterator.GoToBegin();
        while(!vesselimageIterator.IsAtEnd())
        {

             if (vesselimageIterator.Get() > outimageIterator.Get())
                outimageIterator.Set( vesselimageIterator.Get() );

            ++outimageIterator;
            ++vesselimageIterator;
        }
    }

    if (useMask)
    {
        itk::ImageRegionConstIterator<InputImageType> maskIterator(mask_image,maxImage->GetLargestPossibleRegion());
        outimageIterator.GoToBegin();
        while(!outimageIterator.IsAtEnd())          //Apply brain mask
        {
            if  (maskIterator.Get() == 0)
                outimageIterator.Set(0);
            ++outimageIterator;
            ++maskIterator;
        }
    } // end of vesselextractor.cpp
    else if (isCT)
    {
         itk::ImageRegionIterator<InputImageType> inimageIterator(in_image,maxImage->GetLargestPossibleRegion());
         bool neg_img = false;
         while(!inimageIterator.IsAtEnd() && !neg_img)
         {
             if (inimageIterator.Get() < 0)
             {
                 neg_img = true;
             }
             ++inimageIterator;
         }

         inimageIterator.GoToBegin();
         outimageIterator.GoToBegin();
         while(!inimageIterator.IsAtEnd())
         {
             if (neg_img)
             {
                 if (inimageIterator.Get() >= 1000 || inimageIterator.Get() < 0)
                     outimageIterator.Set(0);
             }
             else
             {
                 if (inimageIterator.Get() >= 2000)
                     outimageIterator.Set(0);
             }
             ++inimageIterator;
             ++outimageIterator;
         }
    }

    if (isBin)
    {
        VesselRescalerType::Pointer vesselrescaler = VesselRescalerType::New();
        InputRescalerType::Pointer inputrescaler = InputRescalerType::New();

        vesselrescaler->SetInput( maxImage );
        vesselrescaler->SetOutputMaximum(1);
        vesselrescaler->SetOutputMinimum(0);
        vesselrescaler->Update();
        VesselImageType::Pointer vessel_image = vesselrescaler->GetOutput();

        inputrescaler->SetInput( in_image );
        inputrescaler->SetOutputMaximum(1);
        inputrescaler->SetOutputMinimum(0);
        inputrescaler->Update();

        VesselImageType::Pointer inres_image = inputrescaler->GetOutput();

        OutputImageType::RegionType region;
        OutputImageType::IndexType start;
        start.Fill(0);

        region.SetSize(in_image->GetLargestPossibleRegion().GetSize());
        region.SetIndex(start);

        VesselImageType::Pointer out_image = VesselImageType::New();
        out_image->SetRegions(region);
        out_image->Allocate();
        out_image->SetSpacing( in_image->GetSpacing());
        out_image->SetOrigin( in_image->GetOrigin() );
        out_image->SetDirection( in_image->GetDirection() );

        itk::ImageRegionConstIterator<VesselImageType> imageIterator(inres_image,maxImage->GetLargestPossibleRegion());
        itk::ImageRegionConstIterator<VesselImageType> vesselimageIterator(vessel_image,maxImage->GetLargestPossibleRegion());
        itk::ImageRegionIterator<VesselImageType> outimageIterator(out_image,maxImage->GetLargestPossibleRegion());

        while(!imageIterator.IsAtEnd())
        {
            if  (vesselimageIterator.Get() != 0)
            {
                // Get the value of the current pixel
                InternalPixelType val_in = vesselimageIterator.Get();
                InternalPixelType im_in = imageIterator.Get();

                float val_out = 0;

                //if (im_in < 0.9765625)// try to remove bones
                    val_out = ((double) im_in) * ((double) val_in);

                outimageIterator.Set( val_out ); //static_cast<OutputPixelType>(final_out));
            }
            else
            {
                outimageIterator.Set( 0 );
            }
            ++imageIterator;
            ++vesselimageIterator;
            ++outimageIterator;


        }

        VesselRescalerType::Pointer vesselrescaler2 = VesselRescalerType::New();

        vesselrescaler2->SetInput( out_image );
        vesselrescaler2->SetOutputMaximum(1);
        vesselrescaler2->SetOutputMinimum(0);
        vesselrescaler2->Update();

        typedef itk::BinaryThresholdImageFilter< VesselImageType, OutputImageType > ThresholdFilter;

        ThresholdFilter::Pointer threshfilter = ThresholdFilter::New();
        threshfilter->SetInput( vesselrescaler2->GetOutput() );
        threshfilter->SetLowerThreshold( 0.005 );
        threshfilter->SetUpperThreshold( 0.3 );
        threshfilter->SetOutsideValue( 0 );
        threshfilter->SetInsideValue( 255 );
        threshfilter->Update();

        typedef itk::ConnectedComponentImageFilter <OutputImageType, OutputImageType >
            ConnectedComponentImageFilterType;

          ConnectedComponentImageFilterType::Pointer labelFilter
            = ConnectedComponentImageFilterType::New ();
          labelFilter->SetInput(threshfilter->GetOutput());
          labelFilter->Update();

          itk::ImageRegionIterator<OutputImageType> labimageIterator(labelFilter->GetOutput(),out_image->GetLargestPossibleRegion());
          std::vector<unsigned int> num_elems(std::numeric_limits<OutputPixelType>::max(), 0);

          labimageIterator.GoToBegin();
          while(!labimageIterator.IsAtEnd())
          {
              if (labimageIterator.Get() != 0)
                  num_elems[labimageIterator.Get()-1]++;
              ++labimageIterator;
          }

          OutputPixelType max_elems = 0, min_elems = std::numeric_limits<unsigned int>::max();
          unsigned int max_indx = -1, min_indx = -1;
          for (size_t i = 0; i < std::numeric_limits<OutputPixelType>::max(); ++i)
          {
              if (num_elems[i] > max_elems)
              {
                  max_indx = i;
                  max_elems = num_elems[i];
              }
              if (num_elems[i] < min_elems && num_elems[i] != 0)
              {
                  min_indx = i;
                  min_elems = num_elems[i];
              }

          }

          max_indx++;
          min_indx++;
           OutputPixelType thres_vol = static_cast<OutputPixelType>(floor(0.02 * max_elems) );

         /* std::cout << "Max volume for label: " << max_indx << " volume: " << max_elems << std::endl;
          std::cout << "Min volume for label: " << min_indx << " volume: " << min_elems << std::endl;
          std::cout << "Min permitted volume: " << thres_vol << std::endl; */

          OutputImageType::Pointer final_image = OutputImageType::New();
          final_image->SetRegions(region);
          final_image->Allocate();
          final_image->SetSpacing( reader->GetOutput()->GetSpacing());
          final_image->SetDirection( reader->GetOutput()->GetDirection() );
          final_image->SetOrigin( reader->GetOutput()->GetOrigin() );

          itk::ImageRegionIterator<OutputImageType> finalimageIterator(final_image,out_image->GetLargestPossibleRegion());
          finalimageIterator.GoToBegin();
          labimageIterator.GoToBegin();


          while(!finalimageIterator.IsAtEnd())
          {
              OutputPixelType lab_vol = num_elems[labimageIterator.Get()-1];

              if (lab_vol >= thres_vol)
                  finalimageIterator.Set(255);
              else
                  finalimageIterator.Set(0);

              ++finalimageIterator;
              ++labimageIterator;
          }

        
       
        writer->SetInput( final_image ) ; //out_image );
            
    }
    else
    {
        CastOutFilterType::Pointer caster = CastOutFilterType::New();
        caster->SetInput( maxImage ); 
        caster->Update();
        writer->SetInput( caster->GetOutput() );
    }

    writer->SetFileName( outputImageName );		
    
        
    try
    {
	writer->Update();
    }
    catch( itk::ExceptionObject & err )
    {
	    std::cerr << "Failed: " << err << std::endl;
	    return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;   
}
