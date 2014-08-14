/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <limits>
#include <mitkOpenCVPointTypes.h>
#include <mitkOpenCVFileIOUtils.h>
#include <niftkFileHelper.h>
#include <niftkVideoToImagesCLP.h>
#include <mitkExceptionMacro.h>

#include <fstream>
int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    std::vector <std::string> videoFiles = niftk::FindVideoData(videoInputDirectory);
    std::vector <std::string> frameMapFiles = mitk::FindVideoFrameMapFiles(videoInputDirectory);

    if ( videoFiles.size() > 1 )
    {
      MITK_WARN << "Found multiple video files, will only use " << videoFiles[0];
    }
    if ( frameMapFiles.size() > 1 )
    {
      MITK_WARN << "Found multiple framemap files, will only use " << frameMapFiles[0];
    }

    if ( videoFiles.size() == 0 || frameMapFiles.size() == 0 ) 
    {
      MITK_ERROR << "Failed to find video and/or frameMapFiles in directory " << videoInputDirectory;
      exit(1);
    }

    cv::VideoCapture* capture;
    
    try 
    {
      capture = mitk::InitialiseVideoCapture(videoFiles[0], ignoreVideoReadFail);
    }
    catch (std::exception& e)
    {
      MITK_ERROR << "Caught std::exception:" << e.what();
      returnStatus = -1;
      return returnStatus;
    }
    catch (...)
    {
      MITK_ERROR << "Caught unknown exception:";
      returnStatus = -2;
      return returnStatus;
    }


    std::ifstream* fin = new std::ifstream(frameMapFiles[0].c_str());

    if ( (! capture) || (!fin) )
    {
      MITK_ERROR << "Failed to open video and/or frameMapFile";
      exit(1);
    }
    
    unsigned int framecount = 0 ;
    if ( framesToUse < 0 ) 
    {
      framesToUse = std::numeric_limits<double>::infinity();
    }

    while(framecount < framesToUse )
    {
      mitk::VideoFrame frame;
      try
      {
        frame = mitk::VideoFrame(capture, fin);
        MITK_INFO << "Writing frame " << framecount;
      }
      catch (std::exception& e)
      {
        MITK_ERROR << "Caught exception:" << e.what();
        break;
      }
      frame.WriteToFile(outputPrefix);
      
      framecount ++;
    }

    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }

  return returnStatus;
}
