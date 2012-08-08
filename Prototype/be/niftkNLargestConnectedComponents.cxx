/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

/*!
 * \file niftkConnectedComponents.cxx
 * \page niftkConnectedComponents
 * \section niftkConnectedComponentsSummary Runs ITK ConnectedComponentImageFilter to find connected components.
 *
 * This program runs ITK ConnectedComponentImageFilter to find connected components.
 * \li Dimensions: 3
 * \li Pixel type: Scalars only, of unsigned char, char, short. 
 *
 */
void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Runs ITK ConnectedComponentImageFilter to find connected components. " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " <input> <output_prefix> <output_ext> -largest -background background -verbose" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "  <input>            Input file" << std::endl;
  std::cout << "  <output_prefix>    Output file prefix" << std::endl;
  std::cout << "  <output_ext>       Output file extension" << std::endl;
  std::cout << std::endl;
  std::cout << "*** [optional] ***" << std::endl << std::endl;
  std::cout << "  <-nlargest>        Specifiy to only save the n largest components. [-1]  All components are saved by default." << std::endl;
  std::cout << "  <-background>      Specifiy the background value of the output image. [0]" << std::endl;
  std::cout << "  <-foreground>      Specifiy the foreground value of the output image. Defaults to label value." << std::endl;
  std::cout << "  <-verbose>         More output. No by default" << std::endl;
  std::cout << std::endl;
}



// Sort pairs in descending order, thus largest elements first
template<class T>
struct larger_second
: std::binary_function<T,T,bool>
{
   inline bool operator()(const T& lhs, const T& rhs)
   {
      return lhs.second > rhs.second;
   }
};




int main(int argc, char** argv)
{
  const unsigned int Dimension = 3;
  typedef short InputPixelType;
  typedef short OutputPixelType;
  
  std::string    inputImageName; 
  std::string    outputImagePrefixName; 
  std::string    outputImageExtName; 
  int            iNLargestComps         = -1;
  //bool           isOnlySaveLargest      = false; 
  InputPixelType backgroundValue        =    0; 
  InputPixelType foregroundValue        =  255; 
  bool           foregroundValueSet     = false;
  bool           isVerbose              = false; 
  bool           isFullyConnected       = false;

  if (argc < 3)
  {
    StartUsage(argv[0]); 
    return EXIT_FAILURE; 
  }

  int argIndex   = 1; 
  inputImageName = argv[argIndex];
  std::cout << "input=" << inputImageName<< std::endl;
  
  argIndex += 1; 
  outputImagePrefixName = argv[argIndex]; 
  std::cout << "output_prefix=" << outputImagePrefixName<< std::endl;
  
  argIndex += 1; 
  outputImageExtName = argv[argIndex]; 
  std::cout << "output_ext=" << outputImageExtName<< std::endl;

  for (int i=4; i < argc; i++)
  {
    if (strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0)
    {
      StartUsage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-fullyconnected") == 0)
    {
      isFullyConnected = true; 
      std::cout << "Set -fullyconnected"<< std::endl;
    }
    else if( strcmp( argv[i], "-nlargest" ) == 0 )
    {
      iNLargestComps = atoi( argv[++i] ); 
      std::cout << "Set -nlargest=" << iNLargestComps << std::endl;
    }
    else if(strcmp(argv[i], "-background") == 0)
    {
      backgroundValue = atoi(argv[++i]);
      std::cout << "Set -backgroundValue=" << niftk::ConvertToString(backgroundValue)<< std::endl;
    }
	else if(strcmp(argv[i], "-foreground") == 0)
    {
		foregroundValue = atoi(argv[++i]);
		foregroundValueSet = true;
        std::cout << "Set -foregroundValue=" << niftk::ConvertToString(backgroundValue)<< std::endl;
    }
    
    else if(strcmp(argv[i], "-verbose") == 0)
    {
      isVerbose = true; 
      std::cout << "Set -verbose"<< std::endl;
    }
    else 
    {
      std::cout << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      StartUsage(argv[0]);
      return -1;
    }            
  }
  

  typedef itk::Image< InputPixelType,  Dimension > InputImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  
  typedef itk::ImageFileReader< InputImageType >   ImageFileReaderType;
  typedef itk::ImageFileWriter< InputImageType >   ImageFileWriterType;
  
  typedef itk::ConnectedComponentImageFilter< InputImageType, OutputImageType > ConnectedComponentImageFilterType;
  
  try
  {
    ImageFileReaderType::Pointer reader = ImageFileReaderType::New(); 
    reader->SetFileName( inputImageName ); 
    reader->Update(); 
  
    ConnectedComponentImageFilterType::Pointer ccFilter = ConnectedComponentImageFilterType::New();
    ccFilter->SetInput( reader->GetOutput() );
    ccFilter->SetFullyConnected( isFullyConnected );
    ccFilter->SetBackgroundValue( backgroundValue ); 
    ccFilter->Update(); 
    
    ConnectedComponentImageFilterType::LabelType numberOfComponents = ccFilter->GetObjectCount(); 
    std::cout << "Number of connected components found=" << numberOfComponents << std::endl; 
    
    // Count the number of voxels in each components.
    std::map< OutputPixelType, double > componentSizes; 
    
	typedef itk::ImageRegionConstIterator<OutputImageType> ImageRegionConstIteratorType;
    ImageRegionConstIteratorType ccIt( ccFilter->GetOutput(), ccFilter->GetOutput()->GetLargestPossibleRegion() );
    
	for ( ccIt.GoToBegin();  !ccIt.IsAtEnd();  ++ccIt )
    {
      if (ccIt.Get() != backgroundValue)
        componentSizes[ccIt.Get()]++; 
    }

	// Find the largest component:
	// Convert map into vector of pairs and sort
	typedef std::pair<OutputPixelType, double> PairType;

	std::vector<PairType> vComponentSizes( componentSizes.begin(), componentSizes.end() );
	std::sort( vComponentSizes.begin(), vComponentSizes.end(), larger_second< PairType >() );
	
	std::cout << "Largest label=" << vComponentSizes.front().first << " with size=" << vComponentSizes.front().second;
    
    // Save specified number of components. Write over the reader. 
    ImageFileWriterType::Pointer writer = ImageFileWriterType::New();  
    
	typedef itk::ImageRegionIterator<InputImageType> ImageRegionIterator;
    ImageRegionIterator outputIt( reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion() );
	
	if (iNLargestComps < 0)
		iNLargestComps = int( vComponentSizes.size() );
	
	for (int i = 0;  i < iNLargestComps;  i++)
	{
        InputPixelType fg = vComponentSizes[i].first;

		if (foregroundValueSet)
			fg = foregroundValue;

	    for (ccIt.GoToBegin(), outputIt.GoToBegin();  !ccIt.IsAtEnd();  ++ccIt, ++outputIt)
        {
			if ( ccIt.Get() == vComponentSizes[i].first )
				outputIt.Set( fg ); 
			else
				outputIt.Set( backgroundValue ); 
		}

        writer->SetInput( reader->GetOutput() ); 
        std::string outputImageName = outputImagePrefixName + niftk::ConvertToString( i ) + "." + outputImageExtName; 
      
	    writer->SetFileName( outputImageName ); 
        writer->Update(); 
    }
  } 

  catch (itk::ExceptionObject &e)
  {
    std::cout << "Exception caught:" << e << std::endl;
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS; 
}


