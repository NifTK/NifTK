/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <iostream>
#include <fstream>
#include <SequentialCpuQds.h>
#include <opencv2/core/types_c.h>
#include <opencv/highgui.h>
#include <mitkLogMacros.h>
#include <niftkImageFeatureMatchingCLP.h>


int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (leftImageFileName.empty() || rightImageFileName.empty())
  {
    MITK_ERROR << "Need both left and right images";
  }

  IplImage* left  = cvLoadImage(leftImageFileName.c_str());
  IplImage* right = cvLoadImage(rightImageFileName.c_str());

  if ((left->width != right->width) ||
      (left->height != right->height))
  {
    std::cerr << "Error: image dimensions differ: "
              << left->width  << 'x' << left->height << " vs "
              << right->width << 'x' << right->height
              << std::endl;
    return 2;
  }

  try
  {
    niftk::SequentialCpuQds  c(left->width, left->height);

    for (int i = 0; i < 1; ++i)
    {
      c.Process(left, right);

      //float t = c.GetFrameProcessingTime();
      //std::cerr << "Took " << t << " ms" << std::endl;
    }

    // this is debug related output. disable for now.
#if 0
    for (int l = 0; l < c.GetNumberOfPyramidLevels(); ++l)
    {
      IplImage* p1 = c.ReadbackPyramidLevel(l, false);
      IplImage* p2 = c.ReadbackPyramidLevel(l, true);
      IplImage* t  = c.ReadbackThreadMap(l);
      IplImage* r1 = c.ReadbackRefmap(l, false);
      IplImage* r2 = c.ReadbackRefmap(l, true);
      IplImage* d  = c.CreateDisparityImage(l);

      std::ostringstream  fn1;
      fn1 << argv[1] << "-l=" << l << ".png";
      cvSaveImage(fn1.str().c_str(), p1);

      std::ostringstream  fn2;
      fn2 << argv[2] << "-l=" << l << ".png";
      cvSaveImage(fn2.str().c_str(), p2);

      std::ostringstream  fn3;
      fn3 << argv[1] << "-threadmap-l=" << l << ".png";
      cvSaveImage(fn3.str().c_str(), t);

      std::ostringstream  fn4;
      fn4 << argv[1] << "-refmap-l=" << l << ".png";
      cvSaveImage(fn4.str().c_str(), r1);

      std::ostringstream  fn5;
      fn5 << argv[2] << "-refmap-l=" << l << ".png";
      cvSaveImage(fn5.str().c_str(), r2);

      std::ostringstream  fn6;
      fn6 << argv[1] << "-disparity-l=" << l << ".png";
      cvSaveImage(fn6.str().c_str(), d);

      cvReleaseImage(&p1);
      cvReleaseImage(&p2);
      cvReleaseImage(&t);
      cvReleaseImage(&r1);
      cvReleaseImage(&r2);
      cvReleaseImage(&d);
    }
#endif

    IplImage* r = c.CreateDisparityImage();
    cvSaveImage((std::string(outputFileName) + "-disparity.png").c_str(), r);
    cvReleaseImage(&r);

#if 1
    // dump us a text file suitable for validation
    std::ofstream   disparityfile(outputFileName.c_str());
    // niftkTriangulate2DPointPairsTo3D does NOT (yet) skip header lines.
    //disparityfile << "# pixelx pixely dispx dispy" << std::endl;
    for (int y = 0; y < left->height; ++y)
    {
      for (int x = 0; x < left->width; ++x)
      {
        CvPoint p = c.GetMatch(x, y);
        if (p.x != 0)
        {
          // this is a format that niftkTriangulate2DPointPairsTo3D can read.
          disparityfile << x << ' ' << y << ' ' << "    " << p.x << ' ' << p.y << std::endl;
        }
      }
    }
    disparityfile.close();
#endif
  }
  catch (const std::exception& e)
  {
    MITK_ERROR << "Caught exception: " << e.what();
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception!";
  }

  cvReleaseImage(&left);
  cvReleaseImage(&right);

  return 0;
}

