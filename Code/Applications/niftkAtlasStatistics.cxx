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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"

#include <set>
#include <algorithm>

/**
 * \class InvalidImageSizeException
 * \brief Exception class for when an image is deemed to be the wrong size (number of voxels).
 */
class InvalidImageSizeException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Invalid image size exception.";
  }
} myex;

void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Takes an atlas, containing a set of image labels, and for each input image, " << std::endl;
  std::cout << "  and for each region, computes a bunch of statistics, resulting in a comma separated output. " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  The input images MUST already be in atlas space, no checking is performed." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -a atlas [options] image1 image2 ... imageN" << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -a <filename>                  Atlas image " << std::endl << std::endl;
  std::cout << "    and at least one of the output options below " << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -count                         Output number of voxels in region" << std::endl;
  std::cout << "    -min                           Output minimum of region" << std::endl;
  std::cout << "    -max                           Output maximum of region" << std::endl;
  std::cout << "    -mean                          Output mean of region" << std::endl;
  std::cout << "    -sd                            Output standard deviation of region" << std::endl;
  std::cout << "    -median                        Output median of region" << std::endl;
  std::cout << "    -iqm                           Output inter quartile mean of region" << std::endl << std::endl;
  std::cout << "    -noname                        Don't output image name " << std::endl;
  std::cout << "    -noheader                      Don't output header" << std::endl;
  std::cout << "    -regions 1,2,3,4,5             Comma separated list of region numbers" << std::endl;
  std::cout << "    -atlasbg <int>                 Background value in atlas image. Default 0." << std::endl;
  std::cout << "    -databg  <float>               Background value in data images. Default 0." << std::endl;
  std::cout << "    -debug                         Turn on debugging" << std::endl;
}

struct arguments
{
  std::string atlasImage;
  std::string regionNumbers;
  int firstArgumentAfterOptions;
  int atlasBackgroundValue;
  double dataBackgroundValue;
  bool outputImageName;
  bool outputHeader;
  bool outputCount;
  bool outputMin;
  bool outputMax;
  bool outputMean;
  bool outputSD;
  bool outputMedian;
  bool outputIQM;
  bool debugging;
  int argc;
  char **argv;
};

template <int Dimension> 
int DoMain(arguments args)
{
  typedef int AtlasDataType;
  typedef float DataType;
  typedef typename std::set<AtlasDataType> SetType;
  typedef typename SetType::const_iterator SetIteratorType;
  
  typedef typename itk::Image< AtlasDataType, Dimension >       AtlasImageType; 
  typedef typename itk::Image< DataType, Dimension >            DataImageType;
  typedef typename itk::ImageFileReader< AtlasImageType  >      AtlasImageReaderType;
  typedef typename itk::ImageFileReader< DataImageType  >       DataImageReaderType;

  // Convert the requested regions to a set.
  std::set<AtlasDataType> setOfRequestedRegions;
  char * tok = strtok(const_cast<char*>(args.regionNumbers.c_str()), ",");
  while (tok != NULL) {
    setOfRequestedRegions.insert(atoi(tok));
    tok = strtok(NULL,",");
  }
  
  typename AtlasImageReaderType::Pointer atlasReader  = AtlasImageReaderType::New();
  atlasReader->SetFileName(  args.atlasImage );
  
  
  try 
    { 
      std::cout << "Loading atlas image:" << args.atlasImage<< std::endl;
      atlasReader->Update();
      std::cout << "Done"<< std::endl;
      
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // First work out how many regions atlas contains, ignoring background value.
  // This is a slow implementation, but will do for now.
  
  SetType atlasLabels;
  SetIteratorType atlasLabelsIterator;
  
  typename AtlasImageType::Pointer atlasImage = atlasReader->GetOutput();
  typename itk::ImageRegionConstIterator<AtlasImageType> atlasImageIterator(atlasImage, atlasImage->GetLargestPossibleRegion());
  
  AtlasDataType atlasValue = 0;
  DataType dataValue = 0;
  
  for (atlasImageIterator.GoToBegin();
       !atlasImageIterator.IsAtEnd();
       ++atlasImageIterator)
    {
      atlasValue = atlasImageIterator.Get();
      
      if (atlasValue != args.atlasBackgroundValue 
          && atlasLabels.find(atlasValue) == atlasLabels.end()
          )
        {
          if (setOfRequestedRegions.size() == 0)
            {
              atlasLabels.insert(atlasValue);        
            }
          else
            {
              if (setOfRequestedRegions.find(atlasValue) != setOfRequestedRegions.end())
                {
                  atlasLabels.insert(atlasValue);      
                }
            }
        }
    }
  
  std::cout << "Got " << niftk::ConvertToString((int)atlasLabels.size()) << " atlas labels"<< std::endl;
  
  typename DataImageReaderType::Pointer imageReader = DataImageReaderType::New();
  
  if (args.outputHeader)
    {
      if (args.outputImageName)
        {
          std::cout << "Filename,";
        }
      
      for (atlasLabelsIterator = atlasLabels.begin(); atlasLabelsIterator != atlasLabels.end(); atlasLabelsIterator++)
        {
          atlasValue = *atlasLabelsIterator;
          
          if (args.outputCount)
            {
              std::cout << "Region-" << atlasValue << "-count,";    
            }
          if (args.outputMin)
            {
              std::cout << "Region-" << atlasValue << "-min,";     
            }
          if (args.outputMax)
            {
              std::cout << "Region-" << atlasValue << "-max,";         
            }
          if (args.outputMean)
            {
              std::cout << "Region-" << atlasValue << "-mean,";         
            }
          if (args.outputSD)
            {
              std::cout << "Region-" << atlasValue << "-sd,";         
            }
          if (args.outputMedian)
            {
              std::cout << "Region-" << atlasValue << "-median,";    
            }
          if (args.outputIQM)
            {
              std::cout << "Region-" << atlasValue << "-iqm,";    
            }
        }
      std::cout << std::endl;
    }

  for (int i = args.firstArgumentAfterOptions; i < args.argc; i++)
    {
      
      try 
        { 
          
          std::string filename = std::string(args.argv[i]);
          imageReader->SetFileName(filename);

          std::cout << "Loading image:" << filename<< std::endl;
          imageReader->Update();
          std::cout << "Done"<< std::endl;

          if (args.outputImageName)
            {
              std::cout << filename << ",";
            }
          
          // Check size.
          if (imageReader->GetOutput()->GetLargestPossibleRegion().GetSize() != 
              atlasReader->GetOutput()->GetLargestPossibleRegion().GetSize())
            {
              std::cerr << "Image size " << imageReader->GetOutput()->GetLargestPossibleRegion().GetSize() \
                << ", doesn't match atlas size " << atlasReader->GetOutput()->GetLargestPossibleRegion().GetSize() \
                << std::endl;
              throw myex;              
            }

          // Really slow implementation, but we aren't in a hurry.
          for (atlasLabelsIterator = atlasLabels.begin(); atlasLabelsIterator != atlasLabels.end(); atlasLabelsIterator++)
            {
              atlasValue = *atlasLabelsIterator;

              double min = std::numeric_limits<double>::max();
              double max = std::numeric_limits<double>::min();
              double mean = 0;
              double squares = 0;
              double stdDev = 0;
              double median = 0;
              double iqm = 0;
              
              unsigned long int counter = 0;
              
              std::vector<DataType> list;
              list.clear();
              
              itk::ImageRegionConstIterator<DataImageType> imageIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
              itk::ImageRegionConstIterator<AtlasImageType> atlasIterator(atlasReader->GetOutput(), atlasReader->GetOutput()->GetLargestPossibleRegion());
              
              for (imageIterator.GoToBegin(), 
                   atlasIterator.GoToBegin();
                   !imageIterator.IsAtEnd();
                   ++imageIterator,
                   ++atlasIterator)
                {
                  if (atlasIterator.Get() == atlasValue)
                    {
                      dataValue = imageIterator.Get();
                      
                      if (fabs(dataValue - args.dataBackgroundValue) > 0.0000001)
                        {
                          if (dataValue < min)
                            {
                              min = dataValue;
                            }
                          else if (dataValue > max)
                            {
                              max = dataValue;
                            }
                          counter++;
                          mean += dataValue;
                          squares += (dataValue*dataValue);
                          list.push_back(dataValue);                          
                        }
                    }
                }
              
              if (counter > 0)
                {
                  mean /= (double)(counter);
                  stdDev = sqrt(((1.0/((double)counter))*squares) - (mean*mean));                  
                }
              else
                {
                  mean = 0;
                  stdDev = 0;
                  min = 0;
                  max = 0;
                  median = 0;
                  iqm = 0;
                }
              
              int size = list.size();
              sort(list.begin(), list.end());

              if (size > 0)
                {
                  if (size % 2 == 0)
                    {
                      int medianUpper = (size-1)/2;
                      int medianLower = medianUpper + 1;
                      
                      median = (list[medianUpper] + list[medianLower])/2.0;
                      
                    }
                  else
                    {
                      median = list[size/2];
                      
                    }
                  
                  int lowerQuartile = (unsigned int)((size-1) * 0.25);
                  int upperQuartile = (unsigned int)((size-1) * 0.75);
                  
                  iqm = 0;
                  for (int j = lowerQuartile; j <= upperQuartile; j++)
                    {
                      iqm += list[j];                  
                    }
                  iqm /= (upperQuartile - lowerQuartile + 1);
                }
              
              if (args.outputCount)
                {
                  std::cout << counter << ",";    
                }
              if (args.outputMin)
                {
                  std::cout << min << ",";
                }
              if (args.outputMax)
                {
                  std::cout << max << ",";
                }
              if (args.outputMean)
                {
                  std::cout << mean << ",";         
                }
              if (args.outputSD)
                {
                  std::cout << stdDev << ",";
                }
              if (args.outputMedian)
                {
                  std::cout << median << ",";    
                }
              if (args.outputIQM)
                {
                  std::cout << iqm << ",";    
                }
            }
          std::cout << std::endl;
        } 
      catch( itk::ExceptionObject & err ) 
        { 
          std::cerr << "ExceptionObject caught !";
          std::cerr << err << std::endl; 
          return -2;
        }                

    }
  
  return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.firstArgumentAfterOptions = std::numeric_limits<int>::min();  
  args.outputImageName = true;
  args.outputHeader = true;
  args.atlasBackgroundValue = 0;
  args.dataBackgroundValue = 0;
  args.outputCount = false;
  args.outputMin = false;
  args.outputMax = false;
  args.outputMean = false;
  args.outputSD = false;
  args.outputMedian = false;
  args.outputIQM = false;
  args.regionNumbers="";
  args.debugging = false;
  
  
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-debug") == 0){
      args.debugging=true;
      std::cout << "Set -debug=" << niftk::ConvertToString(args.debugging)<< std::endl;
    }                
    else if(strcmp(argv[i], "-a") == 0){
      args.atlasImage=argv[++i];
      std::cout << "Set -a=" << args.atlasImage<< std::endl;
    }
    else if(strcmp(argv[i], "-regions") == 0){
      args.regionNumbers=argv[++i];
      std::cout << "Set -regions=" << args.regionNumbers<< std::endl;
    }    
    else if(strcmp(argv[i], "-noname") == 0){
      args.outputImageName=false;
      std::cout << "Set -noname=" << niftk::ConvertToString(args.outputImageName)<< std::endl;
    }
    else if(strcmp(argv[i], "-noheader") == 0){
      args.outputHeader=false;
      std::cout << "Set -noheader=" << niftk::ConvertToString(args.outputHeader)<< std::endl;
    }
    else if(strcmp(argv[i], "-count") == 0){
      args.outputCount=true;
      std::cout << "Set -count=" << niftk::ConvertToString(args.outputCount)<< std::endl;
    }        
    else if(strcmp(argv[i], "-min") == 0){
      args.outputMin=true;
      std::cout << "Set -min=" << niftk::ConvertToString(args.outputMin)<< std::endl;
    }        
    else if(strcmp(argv[i], "-max") == 0){
      args.outputMax=true;
      std::cout << "Set -max=" << niftk::ConvertToString(args.outputMax)<< std::endl;
    }        
    else if(strcmp(argv[i], "-mean") == 0){
      args.outputMean=true;
      std::cout << "Set -mean=" << niftk::ConvertToString(args.outputMean)<< std::endl;
    }        
    else if(strcmp(argv[i], "-sd") == 0){
      args.outputSD=true;
      std::cout << "Set -sd=" << niftk::ConvertToString(args.outputSD)<< std::endl;
    }        
    else if(strcmp(argv[i], "-median") == 0){
      args.outputMedian=true;
      std::cout << "Set -median=" << niftk::ConvertToString(args.outputMedian)<< std::endl;
    }        
    else if(strcmp(argv[i], "-iqm") == 0){
      args.outputIQM=true;
      std::cout << "Set -iqm=" << niftk::ConvertToString(args.outputIQM)<< std::endl;
    }        
    else if(strcmp(argv[i], "-atlassbg") == 0){
      args.atlasBackgroundValue=atoi(argv[++i]);
      std::cout << "Set -atlasbg=" << niftk::ConvertToString(args.atlasBackgroundValue)<< std::endl;
    }
    else if(strcmp(argv[i], "-databg") == 0){
      args.dataBackgroundValue=atoi(argv[++i]);
      std::cout << "Set -databg=" << niftk::ConvertToString(args.dataBackgroundValue)<< std::endl;
    }        
    else 
    {
      if(args.firstArgumentAfterOptions < 0)
        {
          args.firstArgumentAfterOptions = i;
          std::cout << "First agument, assumed to be an image=" << niftk::ConvertToString((int)args.firstArgumentAfterOptions)<< std::endl;
          break;
        }    
    }      
  } // end for

  if (argc == 1)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  if (argc < 3)
    {
      std::cerr << argv[0] << ":\tYou should specify at least 2 images" << std::endl;
      return EXIT_FAILURE;
    }

  if (args.atlasImage.length() == 0)
    {
      std::cerr << argv[0] << ":\tYou didn't specify an atlas image" << std::endl;
      return EXIT_FAILURE;
    }
  
  if (args.firstArgumentAfterOptions < 0)
    {
      std::cerr << argv[0] << ":\tYou didn't specify an image to test" << std::endl;
      return EXIT_FAILURE;
    }

  if (args.outputCount == false 
      && args.outputIQM == false 
      && args.outputMax == false 
      && args.outputMean == false 
      && args.outputMedian == false 
      && args.outputMin == false 
      && args.outputSD == false )
    {
      std::cerr << argv[0] << ":\tYou must specify at least one of the output options -count, -min, -max, -mean, -sd, -median or -iqm" << std::endl;
      return EXIT_FAILURE;
    }
  
  args.argc = argc;
  args.argv = argv;
  
  int dims = itk::PeekAtImageDimension(args.atlasImage);
  int result;
  
  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;

}
