/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <PointClouds/MergePointCloudsWrapper.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <MergePointCloudsCLP.h>


int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;


  try
  {
    niftk::MergePointCloudsWrapper::Pointer   merger = niftk::MergePointCloudsWrapper::New();

    boost::filesystem::recursive_directory_iterator   end_itr;
    boost::regex    mpsfilter("(.+\\.mps)");

    for (boost::filesystem::recursive_directory_iterator it(pointCloudDirectory); it != end_itr ; ++it)
    {
      if (boost::filesystem::is_regular_file(it->status()))
      {
        boost::cmatch what;
        const std::string filenamecomponent = it->path().filename().string();

        if (boost::regex_match(filenamecomponent.c_str(), what, mpsfilter))
        {
          merger->AddPointSet(it->path().string());
        }
      }
    }

    mitk::PointSet::Pointer   result = merger->GetOutput();

    // dont bother with mitk's pointset writer. it's ridiculously slow.
    std::ofstream   mpsfile(output);
    mpsfile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<point_set_file>\n<file_version>0.1</file_version>\n<point_set><time_series><time_series_id>0</time_series_id>" << std::endl;
    for (mitk::PointSet::PointsConstIterator i = result->Begin(); i != result->End(); ++i)
    {
      const mitk::PointSet::PointType& p = i->Value();
      mpsfile << "<point><id>" << i->Index() << "</id><specification>0</specification>"
              << "<x>" << p[0] << "</x><y>" << p[1] << "</y><z>" << p[2] << "</z></point>" << std::endl;
    }
    mpsfile << "</time_series></point_set></point_set_file>" << std::endl;
    mpsfile.close();
  }
  catch (const std::exception& e)
  {
    MITK_ERROR << "Caught '" << typeid(e).name() << "': " << e.what();
  }
  catch (...)
  {
    MITK_ERROR << "Caught exception!";
  }

  return returnStatus;
}
