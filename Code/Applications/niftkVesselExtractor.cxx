/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-10-09 14:43:44 +0000 (Tue, 9 Oct 2012) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : maria.zuluaga@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkHessian3DToVesselnessMeasureImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkJoinSeriesImageFilter.h"
#include "itkImage.h"
#include "itkFlipImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

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
"     <image fileExtensions=\"*.nii,*.nii.gz,*.mhd\">\n"
"          <name>outputImageName</name>\n"
"          <flag>o</flag>\n"
"          <description>Output image</description>\n"
"          <label>Output Image</label>\n"
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
"         <default>1</default>\n"
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
"         <default>10</default>\n"
"      </float>\n"
"   </parameters>\n"
"</executable>\n";

void Usage(char *exec)
{
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Applies Sato's vesselness filter to an image using a range of scales." << std::endl;
    std::cout << "  " << std::endl;
    std::cout  << "  " << exec << " [-i inputFileName -o outputFileName -ns NumberofScales ] [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input  image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl;
    std::cout << "    --ns    <int>        Number of scales" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    --mod    <int>   [0]     Linear (0) or exponential (1) scale generation" << std::endl;
    std::cout << "    --aone   <float> [0.5]   Alpha one parameter" << std::endl;
    std::cout << "    --atwo   <float> [2.0]   Alpha two parameter" << std::endl;
    std::cout << "    --max    <float> [10]    Maximum scale for exponential mode" << std::endl;
}
  

int main( int argc, char *argv[] )
{
    // Define command line params
    std::string inputImageName;
    std::string outputImageName;
    unsigned int scales = 0;
    unsigned int mod = 0;
    float max = 10;
    float alphaone = 0.5;
    float alphatwo = 2.0;
    
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
	std::cerr << "Unknown scale mode. Muste be 0 (linear) or 1 (exponential)" << std::endl;
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
    const unsigned int TempDim = 4;
    typedef   short    InputPixelType;
    typedef   float    OutputPixelType;
    typedef   double   InternalPixelType;
  
    typedef   itk::Image< InputPixelType, Dimension >   InputImageType;
    typedef   itk::Image< InternalPixelType, Dimension >   InImageType;
    typedef   itk::Image< OutputPixelType, Dimension >  OutputImageType;
    typedef   itk::Image< OutputPixelType, TempDim > HigherDimImageType;
    
    typedef   itk::CastImageFilter< InputImageType, InImageType > CastFilterType;
    typedef   itk::HessianRecursiveGaussianImageFilter< 
                            InImageType >              HessianFilterType;
    typedef   itk::Hessian3DToVesselnessMeasureImageFilter<
              OutputPixelType > VesselnessMeasureFilterType;
    typedef   itk::JoinSeriesImageFilter< OutputImageType, HigherDimImageType > JoinFilterType;
    typedef   itk::ImageFileReader< InputImageType >  ReaderType;
    typedef   itk::ImageFileWriter< OutputImageType > WriterType;
    
   
    ReaderType::Pointer   reader = ReaderType::New();			    			    
    reader->SetFileName( inputImageName );
    reader->Update();
     
    InputImageType::SpacingType spacing = reader->GetOutput()->GetSpacing(); 
    InputImageType::SizeType size_in = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
         
    float min_spacing = static_cast<float>(spacing[0]);
      
     std::vector<float> all_scales(scales,0);
     switch (mod)
     {
	 case 0: 
	 {
	     for (unsigned int s = 0; s < scales; ++s) 
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
    caster->SetInput( reader->GetOutput() ); 
    
    HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
    VesselnessMeasureFilterType::Pointer vesselnessFilter = 
                            VesselnessMeasureFilterType::New();
    JoinFilterType::Pointer joinFilter = JoinFilterType::New();
    
    hessianFilter->SetInput( caster->GetOutput() );
    vesselnessFilter->SetInput( hessianFilter->GetOutput() );
    vesselnessFilter->SetAlpha1( static_cast< double >(alphaone) );
    vesselnessFilter->SetAlpha2( static_cast< double >(alphatwo) );

    OutputImageType::Pointer vesselnessImage;
    for (size_t s = 0; s < all_scales.size(); ++s)
    {
	hessianFilter->SetSigma( static_cast< double >( all_scales[s] ) );
	vesselnessFilter->Update();
	vesselnessImage = vesselnessFilter->GetOutput();
	vesselnessImage->DisconnectPipeline();
	joinFilter->SetInput( s, vesselnessImage );
    }
    
    joinFilter->Update();
    HigherDimImageType::Pointer higher_img = joinFilter->GetOutput();
    HigherDimImageType::SizeType size = higher_img->GetLargestPossibleRegion().GetSize();
    
    OutputImageType::Pointer maxImage = OutputImageType::New();
    OutputImageType::IndexType outind;
    OutputImageType::RegionType region;
    
    outind.Fill(0);
    region.SetSize(size_in);    
    region.SetIndex(outind);
    
    maxImage->SetRegions(region);
    maxImage->SetOrigin( reader->GetOutput()->GetOrigin() );
    maxImage->SetDirection( reader->GetOutput()->GetDirection() );
    maxImage->Allocate();
    maxImage->SetSpacing( spacing );

    itk::Index<TempDim> index;
    itk::Index<Dimension> ind;
    for (index[0] = 0,  ind[0] = 0; ((unsigned int)index[0]) < size[0]; index[0]++, ind[0]++)
	 for (index[1] = 0, ind[1] = 0; ((unsigned int)index[1]) < size[1]; index[1]++, ind[1]++)
	      for (index[2] = 0, ind[2] = 0; ((unsigned int) index[2]) < size[2]; index[2]++, ind[2]++)
	      {
		  float max_val = 0.0, value = 0;
		   for (index[3] = 0; ((unsigned int)index[3]) < size[3]; index[3]++)
		   {
		       value = higher_img->GetPixel(index);
		       if (max_val < value)
			   max_val = value;
		   }
		   maxImage->SetPixel(ind, max_val);
	      }
    
    WriterType::Pointer   writer = WriterType::New();
    writer->SetFileName( outputImageName );		
    writer->SetInput( maxImage );
        
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