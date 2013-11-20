/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FitPlaneToPointCloudWrapper.h"


namespace niftk
{


//-----------------------------------------------------------------------------
FitPlaneToPointCloudWrapper::FitPlaneToPointCloudWrapper()
{
}


//-----------------------------------------------------------------------------
FitPlaneToPointCloudWrapper::~FitPlaneToPointCloudWrapper()
{
}


} // namespace


/*

#include <iostream>
#include <fstream>
#include <mitkPointSetReader.h>
#include <mitkPointSetWriter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <boost/typeof/typeof.hpp>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>


int main(int argc, char* argv[])
{
  if (argc < 1)
  {
    std::cerr << "specify an mitk mps pointset file!" << std::endl;
    return 1;
  }

  std::string   mpsfilename(argv[1]);

  bool  pcdfileexists = false;
  std::ifstream   filecheck((mpsfilename + ".pcd").c_str());
  pcdfileexists = filecheck.good();
  filecheck.close();

  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud1(new pcl::PointCloud<pcl::PointXYZ>);
  if (!pcdfileexists)
  {
    std::cerr << "Reading pointset from file: " << mpsfilename;
    mitk::PointSetReader::Pointer   psreader = mitk::PointSetReader::New();
    psreader->SetFileName(mpsfilename);
    psreader->Update();
    mitk::PointSet::Pointer pointset = psreader->GetOutput();
    std::cerr << "...done" << std::endl;


    std::cerr << "Converting to PCL pointcloud...";
    for (mitk::PointSet::PointsConstIterator i = pointset->Begin(); i != pointset->End(); ++i)
    {
      const mitk::PointSet::PointType& p = i->Value();
      cloud1->push_back(pcl::PointXYZ(p[0], p[1], p[2]));
    }
    std::cerr << "done" << std::endl;

    pcl::io::savePCDFileBinary(mpsfilename + ".pcd", *cloud1);
  }
  else
  {
    std::cerr << "Loading cached PCD file (much faster!)...";
    pcl::io::loadPCDFile(mpsfilename + ".pcd", *cloud1);
    std::cerr << "done" << std::endl;
  }


  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.setInputCloud(cloud1);

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
  }


  std::string   outputfilename(mpsfilename + ".plane.txt");
  std::ofstream   outputfile(outputfilename.c_str());
  outputfile << coefficients->values[0] << ' ' << coefficients->values[1] << ' ' << coefficients->values[2] << ' ' << coefficients->values[3] << std::endl;

  // compute distance of each point to the fitted plane.
  float   minplanedistance =  std::numeric_limits<float>::max();
  float   maxplanedistance = -std::numeric_limits<float>::max();
  float   avgplanedistance = 0;
  float   rmsplanedistance = 0;
  float   planecoeffthingy = std::sqrt(coefficients->values[0] * coefficients->values[0] +  coefficients->values[1] * coefficients->values[1] + coefficients->values[2] * coefficients->values[2]);
  for (pcl::PointCloud<pcl::PointXYZ>::const_iterator i = cloud1->begin(); i != cloud1->end(); ++i)
  {
    float   dist = std::abs(coefficients->values[0] * i->x + coefficients->values[1] * i->y + coefficients->values[2] * i->z + coefficients->values[3]);
    dist /= planecoeffthingy;

    rmsplanedistance += dist * dist;
    avgplanedistance += dist;
    minplanedistance = std::min(minplanedistance, dist);
    maxplanedistance = std::max(maxplanedistance, dist);
  }
  avgplanedistance /= cloud1->size();
  rmsplanedistance = std::sqrt(rmsplanedistance / cloud1->size());
  outputfile 
    << "# minimum distance to estimated plane: " << minplanedistance << std::endl
    << "# maximum distance to estimated plane: " << maxplanedistance << std::endl
    << "# average distance to estimated plane: " << avgplanedistance << std::endl
    << "# rms distance to estimated plane: " << rmsplanedistance << std::endl
    ;

  outputfile.close();
}
*/