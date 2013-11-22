/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <PointClouds/MergePointCloudsWrapper.h>


int main(int argc, char* argv[])
{
  if (argc < 1)
  {
    std::cerr << "specify one or more mitk mps pointset file(s)!" << std::endl;
    return 1;
  }

  /*

  std::string   firstfilename(argv[1]);

  unsigned int    id = 0;
  mitk::PointSet::Pointer   output = mitk::PointSet::New();
  for (int i = 1; i < argc; ++i)
  {
    std::string   mpsfilename(argv[i]);
    std::cerr << "Reading pointset from file: " << mpsfilename;

    mitk::PointSetReader::Pointer   psreader = mitk::PointSetReader::New();
    psreader->SetFileName(mpsfilename);
    psreader->Update();
    mitk::PointSet::Pointer pointset = psreader->GetOutput();
    std::cerr << "...done" << std::endl;

    for (mitk::PointSet::PointsConstIterator i = pointset->Begin(); i != pointset->End(); ++i, ++id)
    {
      const mitk::PointSet::PointType& p = i->Value();
      output->InsertPoint(id, mitk::PointSet::PointType(&p[0]));
    }
  }

  // dont bother with mitk's pointset writer. it's ridiculously slow.
  std::ofstream   mpsfile(firstfilename + ".merged.mps");
  mpsfile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<point_set_file>\n<file_version>0.1</file_version>\n<point_set><time_series><time_series_id>0</time_series_id>" << std::endl;
  for (mitk::PointSet::PointsConstIterator i = output->Begin(); i != output->End(); ++i)
  {
    const mitk::PointSet::PointType& p = i->Value();
    mpsfile << "<point><id>" << i->Index() << "</id><specification>0</specification>"
            << "<x>" << p[0] << "</x><y>" << p[1] << "</y><z>" << p[2] << "</z></point>" << std::endl;
  }
  mpsfile << "</time_series></point_set></point_set_file>" << std::endl;
  mpsfile.close();
  */
}
