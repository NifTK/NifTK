/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVFileIOUtils.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <mitkTimeStampsContainer.h>
#include <niftkFileHelper.h>
#include <boost/math/special_functions/fpclassify.hpp>

namespace mitk {

//---------------------------------------------------------------------------
bool CheckIfDirectoryContainsTrackingMatrices(const std::string& directory)
{
  boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
  boost::filesystem::directory_iterator endItr;
  
  for ( boost::filesystem::directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringthing = it->path().filename().string();
      if ( boost::regex_match( stringthing.c_str(), what, timeStampFilter) )
      {
        return true;
      }
    }
  }
  return false;
}


//---------------------------------------------------------------------------
std::vector<std::string> FindTrackingMatrixDirectories(const std::string& directory)
{
  std::vector<std::string> directories;
  boost::filesystem::recursive_directory_iterator endItr;
  
  for ( boost::filesystem::recursive_directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_directory (it->status()) )
    {
      if ( CheckIfDirectoryContainsTrackingMatrices(it->path().string()))
      {
         directories.push_back(it->path().string());
      }
    }
  }
  std::sort (directories.begin(), directories.end());
  return directories;
}


//---------------------------------------------------------------------------
mitk::TimeStampsContainer FindTrackingTimeStamps(std::string directory)
{
  boost::filesystem::directory_iterator endItr;
  boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
  TimeStampsContainer returnStamps;
  
  for ( boost::filesystem::directory_iterator it(directory);it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringThing = it->path().filename().string();
      if ( boost::regex_match( stringThing.c_str(), what, timeStampFilter) )
      {
        returnStamps.Insert(boost::lexical_cast<unsigned long long>(it->path().filename().stem().string().c_str()));
      }
    }
  }
  returnStamps.Sort();
  return returnStamps;
}


//---------------------------------------------------------------------------
std::vector<std::string> FindVideoFrameMapFiles(const std::string directory)
{
  boost::filesystem::recursive_directory_iterator endItr;
  boost::regex frameLogFilter ( "(.+)(framemap.log)");
  std::vector<std::string> returnStrings;

  for ( boost::filesystem::recursive_directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      const std::string stringThing = it->path().filename().string();

      if ( boost::regex_match( stringThing.c_str(), what, frameLogFilter) )
      {
        returnStrings.push_back(it->path().string());
      }
    }
  }
  return returnStrings;
}


//---------------------------------------------------------------------------
bool ReadTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix)
{
  bool isSuccessful = false;
  if (outputMatrix.rows != 4)
  {
    mitkThrow() << "ReadTrackerMatrix: Matrix does not have 4 rows" << std::endl;
  }
  if (outputMatrix.cols != 4)
  {
    mitkThrow() << "ReadTrackerMatrix: Matrix does not have 4 columns" << std::endl;
  }

  cv::Matx44d matrix;
  isSuccessful = ReadTrackerMatrix(filename, matrix);
  if (isSuccessful)
  {
    for ( int row = 0 ; row < 4 ; row ++ )
    {
      for ( int col = 0 ; col < 4 ; col ++ )
      {
        outputMatrix.at<double>(row,col) = matrix(row, col);
      }
    }
  }
  return isSuccessful;
}


//---------------------------------------------------------------------------
bool ReadTrackerMatrix(const std::string& filename, cv::Matx44d& outputMatrix)
{
  bool isSuccessful = false;
  std::ifstream fin(filename.c_str());
  if ( !fin )
  {
    MITK_WARN << "ReadTrackerMatrix: Failed to open matrix file " << filename;
    return isSuccessful;
  }

  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      fin >> outputMatrix(row, col);
    }
  }
  isSuccessful = true;
  return isSuccessful;
}


//---------------------------------------------------------------------------
bool SaveTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix)
{
  bool isSuccessful = false;
  if (outputMatrix.rows != 4)
  {
    mitkThrow() << "SaveTrackerMatrix: Matrix does not have 4 rows" << std::endl;
  }
  if (outputMatrix.cols != 4)
  {
    mitkThrow() << "SaveTrackerMatrix: Matrix does not have 4 columns" << std::endl;
  }

  cv::Matx44d matrix;
  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      matrix(row, col) = outputMatrix.at<double>(row,col);
    }
  }

  isSuccessful = SaveTrackerMatrix(filename, matrix);
  return isSuccessful;
}


//---------------------------------------------------------------------------
bool SaveTrackerMatrix(const std::string& filename, cv::Matx44d& outputMatrix)
{
  bool isSuccessful = false;
  std::ofstream fout(filename.c_str());
  if ( !fout )
  {
    MITK_WARN << "SaveTrackerMatrix: Failed to open matrix file " << filename;
    return isSuccessful;
  }
  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      fout << outputMatrix(row,col);
      if ( col < 3 )
      {
        fout << " ";
      }
    }
    fout << std::endl;
  }
  fout.close();
  isSuccessful = true;
  return isSuccessful;
}


//---------------------------------------------------------------------------
cv::VideoCapture* InitialiseVideoCapture ( std::string filename , bool ignoreErrors )
{
  cv::VideoCapture* capture = new cv::VideoCapture (filename);
  if ( ! capture )
  {
    mitkThrow() << "Failed to open " << filename << " for video capture" << std::endl;
  }
  //try and get some information about the capture, if these calls fail it may be that 
  //the capture may still work but will exhibit undesirable behaviour, see trac 3718
  int m_VideoWidth = capture->get(CV_CAP_PROP_FRAME_WIDTH);
  int m_VideoHeight = capture->get(CV_CAP_PROP_FRAME_HEIGHT);

  if ( m_VideoWidth == 0 || m_VideoHeight == 0 )
  {
    if ( ! ignoreErrors )
    {
      mitkThrow() << "Problem opening video file for capture. You may want to try rebuilding openCV with ffmpeg support or if available and you're feeling brave over riding video read errors with an ignoreVideoErrors parameter.";
    }
    else 
    {
      MITK_WARN << "mitk::InitialiseVideo detected errors with video file decoding but persevering any way as ignoreErrors is set true";
    }
  }

  return capture;
}


//---------------------------------------------------------------------------
std::vector< std::pair<unsigned long long, cv::Point3d> > LoadTimeStampedPoints(const std::string& directory)
{
  std::vector< std::pair<unsigned long long, cv::Point3d> > timeStampedPoints;

  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(directory);
  std::sort(pointFiles.begin(), pointFiles.end());

  for (unsigned int i = 0; i < pointFiles.size(); i++)
  {
    cv::Point3d point;
    std::string fileName = pointFiles[i];

    if(fileName.size() > 0)
    {
      std::ifstream myfile(fileName.c_str());
      if (myfile.is_open())
      {
        point.x = 0;
        point.y = 0;
        point.z = 0;

        myfile >> point.x;
        myfile >> point.y;
        myfile >> point.z;

        if (myfile.bad() || myfile.eof() || myfile.fail())
        {
          std::ostringstream errorMessage;
          errorMessage << "Could not load point file:" << fileName << std::endl;
          mitkThrow() << errorMessage.str();
        }
        myfile.close();
      }
    }

    // Parse timestamp.
    boost::regex timeStampFilter ( "([0-9]{19})");
    boost::cmatch what;
    unsigned long long timeStamp = 0;

    if ( boost::regex_match( (niftk::Basename(fileName)).c_str(), what, timeStampFilter) )
    {
      timeStamp = boost::lexical_cast<unsigned long long>(niftk::Basename(fileName));

      if (timeStamp != 0)
      {
        timeStampedPoints.push_back(std::pair<unsigned long long, cv::Point3d>(timeStamp, point));
      }
      else
      {
        std::ostringstream errorMessage;
        errorMessage << "Failed to extract timestamp from name of file:" << fileName << std::endl;
        mitkThrow() << errorMessage.str();
      }
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Could not match timestamp in name of file:" << fileName << std::endl;
      mitkThrow() << errorMessage.str();
    }
  }

  return timeStampedPoints;
}


//---------------------------------------------------------------------------
void LoadTimeStampedPoints(std::vector< std::pair<unsigned long long, cv::Point3d> >& points, 
    std::vector <mitk::ProjectedPointPair>& screenPoints, const std::string& fileName)
{
  if (fileName.length() == 0)
  {
    mitkThrow() << "Filename should not be empty." << std::endl;
  }

  std::ifstream myfile(fileName.c_str());
  if (myfile.is_open())
  {
    cv::Point3d point;

    do
    {
      mitk::TimeStampsContainer::TimeStamp timeStamp = 0;
      double x = 0;
      double y = 0;
      double z = 0;
      double lx = 0;
      double ly = 0;
      double rx = 0;
      double ry = 0;


      myfile >> timeStamp;
      myfile >> x;
      myfile >> y;
      myfile >> z;
      myfile >> lx;
      myfile >> ly;
      myfile >> rx;
      myfile >> ry;
     

      if (timeStamp > 0 && !boost::math::isnan(x) && !boost::math::isnan(y) && !boost::math::isnan(z)) // any other validation?
      {
        point.x = x;
        point.y = y;
        point.z = z;
        points.push_back(std::pair<unsigned long long, cv::Point3d>(timeStamp, point));
        
        mitk::ProjectedPointPair  pointPair( cv::Point2d(lx,ly), cv::Point2d (rx,ry));
        pointPair.SetTimeStamp(timeStamp);
        screenPoints.push_back ( pointPair );
      }
     
    }
    while (!myfile.bad() && !myfile.eof() && !myfile.fail());

    myfile.close();
  }
  else
  {
    mitkThrow() << "Failed to open file " << fileName << " for reading." << std::endl;
  }
}
//---------------------------------------------------------------------------
void LoadTimeStampedPoints(std::vector< std::pair<unsigned long long, cv::Point2d> >& points, 
     const std::string& fileName)
{
  if (fileName.length() == 0)
  {
    mitkThrow() << "Filename should not be empty." << std::endl;
  }

  std::ifstream myfile(fileName.c_str());
  if (myfile.is_open())
  {
    cv::Point2d point;

    do
    {
      mitk::TimeStampsContainer::TimeStamp timeStamp = 0;
      double x = 0;
      double y = 0;

      myfile >> timeStamp;
      myfile >> x;
      myfile >> y;

      if (timeStamp > 0 && !boost::math::isnan(x) && !boost::math::isnan(y) ) // any other validation?
      {
        point.x = x;
        point.y = y;
        points.push_back(std::pair<unsigned long long, cv::Point2d>(timeStamp, point));
      }
     
    }
    while (!myfile.bad() && !myfile.eof() && !myfile.fail());
    myfile.close();
  }
  else
  {
    mitkThrow() << "Failed to open file " << fileName << " for reading." << std::endl;
  }
}



//---------------------------------------------------------------------------
void SaveTimeStampedPoints(const std::vector< std::pair<unsigned long long, cv::Point3d> >& points, const std::string& fileName)
{
  if (fileName.length() == 0)
  {
    mitkThrow() << "Filename should not be empty." << std::endl;
  }

  std::ofstream myfile(fileName.c_str());
  if (myfile.is_open())
  {
    for (unsigned long int i = 0; i < points.size(); i++)
    {
      myfile << points[i].first << " " << points[i].second.x << " " << points[i].second.y << " " << points[i].second.z << std::endl;
    }

    myfile.close();
  }
  else
  {
    mitkThrow() << "Failed to open file " << fileName << " for writing." << std::endl;
  }
}
//---------------------------------------------------------------------------
void SavePickedObjects ( const std::vector < mitk::PickedObject > & points, std::ostream& os )
{
  boost::property_tree::ptree pt;
  pt.add ("picked_object_list.version", 1);
  for ( std::vector<PickedObject>::const_iterator it = points.begin() ; it < points.end() ; ++it )
  {
    if ( it->m_Points.size() != 0 )
    {
      boost::property_tree::ptree& node = pt.add("picked_object_list.picked_object", "");
      node.put("id",it->m_Id);
      node.put("frame",it->m_FrameNumber);
      node.put("channel", it->m_Channel);
      node.put("timestamp",it->m_TimeStamp);

      boost::property_tree::ptree& points = node.add("points", "");
      for ( unsigned int i = 0 ; i < it->m_Points.size() ; i ++ )
      {
        boost::property_tree::ptree& coordinate = points.add("coordinate", "");
        std::ostringstream xyzstream;
        xyzstream << it->m_Points[i].x << " " << it->m_Points[i].y << " " << it->m_Points[i].z; 
        coordinate.put("<xmlattr>.xyz", xyzstream.str());
      }
      if ( it->m_IsLine )
      {
        node.put("<xmlattr>.line",  true);
      }
      else
      {
        node.put("<xmlattr>.line", false);
      }
    }
  }
  boost::property_tree::xml_writer_settings<std::string> settings(' ',2);
  boost::property_tree::write_xml (os, pt, settings);

}

//---------------------------------------------------------------------------
void LoadPickedObjects (  std::vector < mitk::PickedObject > & points, std::istream& is )
{
  boost::property_tree::ptree pt;
  try
    {
    boost::property_tree::read_xml (is, pt);
    BOOST_FOREACH ( boost::property_tree::ptree::value_type const& v , pt.get_child("picked_object_list") )
    {
      MITK_INFO << v.first;
      if ( v.first == "picked_object" )
      {
        mitk::PickedObject po;
        po.m_Id = v.second.get<int> ("id");
        po.m_FrameNumber = v.second.get<unsigned int> ("frame");
        po.m_Channel = v.second.get<std::string> ( "channel" );
        po.m_TimeStamp = v.second.get<unsigned long long > ("timestamp");
        po.m_IsLine = v.second.get<bool> ("<xmlattr>.line", false);
        BOOST_FOREACH ( boost::property_tree::ptree::value_type const& coord , v.second.get_child("points" ) )
        {
           if ( coord.first == "coordinate" )
           {
             std::string xyz = coord.second.get<std::string>("<xmlattr>.xyz", "");
             std::stringstream xyzstream(xyz);
             cv::Point3d point;
             xyzstream >> point.x >> point.y >> point.z;
             po.m_Points.push_back(point);
           }
        }
        points.push_back(po);
      }
    }
  }
  catch(const std::runtime_error& e)
  {
    MITK_ERROR << "Caught " << e.what();
  }             
}



//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadMatricesFromDirectory (const std::string& fullDirectoryName)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  std::sort(files.begin(),files.end(),niftk::NumericStringCompare);
  std::vector<cv::Mat> myMatrices;

  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
      std::ifstream fin(files[i].c_str());
      for ( int row = 0; row < 4; row ++ )
      {
        for ( int col = 0; col < 4; col ++ )
        {
          fin >> Matrix.at<double>(row,col);
        }
      }
      myMatrices.push_back(Matrix);
    }
  }
  else
  {
    mitkThrow() << "No files found in directory!" << std::endl;
  }

  if (myMatrices.size() == 0)
  {
    mitkThrow() << "No Matrices found in directory!" << std::endl;
  }
  std::cout << "Loaded " << myMatrices.size() << " Matrices from " << fullDirectoryName << std::endl;
  return myMatrices;
}


//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadOpenCVMatricesFromDirectory (const std::string& fullDirectoryName)
{
  std::vector<std::string> files = niftk::GetFilesInDirectory(fullDirectoryName);
  std::sort(files.begin(),files.end());
  std::vector<cv::Mat> myMatrices;

  if (files.size() > 0)
  {
    for(unsigned int i = 0; i < files.size();i++)
    {
      if ( niftk::FilenameHasPrefixAndExtension(files[i],"",".extrinsic.xml") )
      {
        cv::Mat Extrinsic = (cv::Mat)cvLoadImage(files[i].c_str());
        if (Extrinsic.rows != 4 )
        {
          mitkThrow() << "Failed to load camera intrinsic params" << std::endl;
        }
        else
        {
          myMatrices.push_back(Extrinsic);
          std::cout << "Loaded: " << Extrinsic << std::endl << "From " << files[i] << std::endl;
        }
      }
    }
  }
  else
  {
    mitkThrow() << "No files found in directory!";
  }

  if (myMatrices.size() == 0)
  {
    mitkThrow() << "No Matrices found in directory!";
  }
  std::cout << "Loaded " << myMatrices.size() << " Matrices from " << fullDirectoryName << std::endl;
  return myMatrices;
}


//-----------------------------------------------------------------------------
std::vector<cv::Mat> LoadMatricesFromExtrinsicFile (const std::string& fullFileName)
{

  std::vector<cv::Mat> myMatrices;
  std::ifstream fin(fullFileName.c_str());

  cv::Mat RotationVector = cvCreateMat(3,1,CV_64FC1);
  cv::Mat TranslationVector = cvCreateMat(3,1,CV_64FC1);
  double temp_d[6];
  while ( fin >> temp_d[0] >> temp_d[1] >> temp_d[2] >> temp_d[3] >> temp_d[4] >> temp_d[5] )
  {
    RotationVector.at<double>(0,0) = temp_d[0];
    RotationVector.at<double>(1,0) = temp_d[1];
    RotationVector.at<double>(2,0) = temp_d[2];
    TranslationVector.at<double>(0,0) = temp_d[3];
    TranslationVector.at<double>(1,0) = temp_d[4];
    TranslationVector.at<double>(2,0) = temp_d[5];

    cv::Mat Matrix = cvCreateMat(4,4,CV_64FC1);
    cv::Mat RotationMatrix = cvCreateMat(3,3,CV_64FC1);
    cv::Rodrigues (RotationVector, RotationMatrix);

    for ( int row = 0; row < 3; row ++ )
    {
      for ( int col = 0; col < 3; col ++ )
      {
        Matrix.at<double>(row,col) = RotationMatrix.at<double>(row,col);
      }
    }

    for ( int row = 0; row < 3; row ++ )
    {
      Matrix.at<double>(row,3) = TranslationVector.at<double>(row,0);
    }
    for ( int col = 0; col < 3; col ++ )
    {
      Matrix.at<double>(3,col) = 0.0;
    }
    Matrix.at<double>(3,3) = 1.0;
    myMatrices.push_back(Matrix);
  }
  return myMatrices;
}


//-----------------------------------------------------------------------------
void LoadStereoCameraParametersFromDirectory (const std::string& directory,
  cv::Mat* leftCameraIntrinsic, cv::Mat* leftCameraDistortion,
  cv::Mat* rightCameraIntrinsic, cv::Mat* rightCameraDistortion,
  cv::Mat* rightToLeftRotationMatrix, cv::Mat* rightToLeftTranslationVector,
  cv::Mat* leftCameraToTracker)
{
  boost::filesystem::directory_iterator end_itr;
  boost::regex leftIntrinsicFilter ("(.+)(left.intrinsic.txt)");
  boost::regex rightIntrinsicFilter ("(.+)(right.intrinsic.txt)");
  boost::regex r2lFilter ("(.+)(r2l.txt)");
  boost::regex handeyeFilter ("(.+)(left.handeye.txt)");

  std::vector<std::string> leftIntrinsicFiles;
  std::vector<std::string> rightIntrinsicFiles;
  std::vector<std::string> r2lFiles;
  std::vector<std::string> handeyeFiles;

  for ( boost::filesystem::directory_iterator it(directory);it != end_itr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      char *  stringthing = new char [512] ;
      strcpy (stringthing,it->path().string().c_str());
      if ( boost::regex_match( stringthing,what,leftIntrinsicFilter) )
      {
        leftIntrinsicFiles.push_back(it->path().string());
      }
      if ( boost::regex_match( stringthing,what,rightIntrinsicFilter) )
      {
        rightIntrinsicFiles.push_back(it->path().string());
      }
      if ( boost::regex_match( stringthing,what,r2lFilter) )
      {
        r2lFiles.push_back(it->path().string());
      }
      if ( boost::regex_match( stringthing,what,handeyeFilter) )
      {
        handeyeFiles.push_back(it->path().string());
      }
    }
  }

  if ( leftIntrinsicFiles.size() != 1 )
  {
    mitkThrow() << "Found the wrong number of left intrinsic files";
  }

  if ( rightIntrinsicFiles.size() != 1 )
  {
    mitkThrow() << "Found the wrong number of right intrinsic files";
  }

  if ( r2lFiles.size() != 1 )
  {
    mitkThrow() << "Found the wrong number of right to left files" << std::endl;
  }

  if ( handeyeFiles.size() != 1 )
  {
    mitkThrow() << "Found the wrong number of handeye files" << std::endl;
  }

  std::cout << "Loading left intrinsics from  " << leftIntrinsicFiles[0] << std::endl;
  LoadCameraIntrinsicsFromPlainText (leftIntrinsicFiles[0],leftCameraIntrinsic, leftCameraDistortion);
  std::cout << "Loading right intrinsics from  " << rightIntrinsicFiles[0] << std::endl;
  LoadCameraIntrinsicsFromPlainText (rightIntrinsicFiles[0],rightCameraIntrinsic, rightCameraDistortion);
  std::cout << "Loading right to left from  " << r2lFiles[0] << std::endl;
  LoadStereoTransformsFromPlainText (r2lFiles[0],rightToLeftRotationMatrix, rightToLeftTranslationVector);
  std::cout << "Loading handeye from  " << handeyeFiles[0] << std::endl;
  LoadHandeyeFromPlainText (handeyeFiles[0],leftCameraToTracker);

}


//-----------------------------------------------------------------------------
void LoadCameraIntrinsicsFromPlainText (const std::string& filename,
    cv::Mat* CameraIntrinsic, cv::Mat* CameraDistortion)
{
  std::ifstream fin(filename.c_str());
  // make sure we throw an exception if parsing fails for any reason.
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
       fin >> CameraIntrinsic->at<double>(row,col);
    }
  }

  if (CameraDistortion != 0)
  {
    // this should work around any row-vs-column vector opencv matrix confusion issues.
    for (int row = 0; row < CameraDistortion->size().height; ++row)
    {
      for (int col = 0; col < CameraDistortion->size().width; ++col)
      {
        fin >> CameraDistortion->at<double>(row, col);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void LoadStereoTransformsFromPlainText (const std::string& filename,
    cv::Mat* rightToLeftRotationMatrix, cv::Mat* rightToLeftTranslationVector)
{
  std::ifstream fin(filename.c_str());
  // make sure we throw an exception if parsing fails for any reason.
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
       fin >> rightToLeftRotationMatrix->at<double>(row,col);
    }
  }

  for (int row = 0; row < rightToLeftTranslationVector->size().height; ++row)
  {
    for (int col = 0; col < rightToLeftTranslationVector->size().width; ++col)
    {
      fin >> rightToLeftTranslationVector->at<double>(row, col);
    }
  }
}


//-----------------------------------------------------------------------------
void LoadHandeyeFromPlainText (const std::string& filename,
    cv::Mat* leftCameraToTracker)
{
  std::ifstream fin(filename.c_str());
  for ( int row = 0; row < 4; row ++ )
  {
    for ( int col = 0; col < 4; col ++ )
    {
       fin >> leftCameraToTracker->at<double>(row,col);
    }
  }

}

} // end namespace
