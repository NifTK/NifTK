/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSurfaceReconstruction.h"
#include "SequentialCpuQds.h"
#include <opencv2/core/core_c.h>
#include <mitkImageReadAccessor.h>


namespace mitk 
{

//-----------------------------------------------------------------------------
SurfaceReconstruction::SurfaceReconstruction()
  : m_SequentialCpuQds(0)
{

}


//-----------------------------------------------------------------------------
SurfaceReconstruction::~SurfaceReconstruction()
{
  delete m_SequentialCpuQds;
}


//-----------------------------------------------------------------------------
void SurfaceReconstruction::Run(const mitk::DataStorage::Pointer dataStorage,
                                mitk::DataNode::Pointer outputNode,
                                const mitk::Image::Pointer image1,
                                const mitk::Image::Pointer image2,
                                Method method)
{
  // sanity check
  assert(dataStorage.IsNotNull());
  assert(image1.IsNotNull());
  assert(image2.IsNotNull());

  int width  = image1->GetDimension(0);
  int height = image1->GetDimension(1);

  // for current methods, both left and right have to have the same size
  if (image2->GetDimension(0) != width)
  {
    throw std::runtime_error("Left and right image width are different");
  }
  if (image2->GetDimension(1) != height)
  {
    throw std::runtime_error("Left and right image height are different");
  }
  // we dont really care here whether the image has a z dimension or not
  // but for debugging purposes might as well check
  assert(image1->GetDimension(2) == 1);
  assert(image2->GetDimension(2) == 1);


  try
  {
    mitk::ImageReadAccessor  leftReadAccess(image1);
    mitk::ImageReadAccessor  rightReadAccess(image2);

    const void* leftPtr = leftReadAccess.GetData();
    const void* rightPtr = rightReadAccess.GetData();

    // FIXME: number of channels???
    int numComponents = 4;

    // mitk images are tightly packed (i hope)
    int bytesPerRow = width * numComponents;

    IplImage  leftIpl;
    cvInitImageHeader(&leftIpl, cvSize(width, height), IPL_DEPTH_8U, numComponents);
    cvSetData(&leftIpl, const_cast<void*>(leftPtr), bytesPerRow);
    IplImage  rightIpl;
    cvInitImageHeader(&rightIpl, cvSize(width, height), IPL_DEPTH_8U, numComponents);
    cvSetData(&rightIpl, const_cast<void*>(rightPtr), bytesPerRow);


    switch (method)
    {
      case SEQUENTIAL_CPU:
      {
        if (m_SequentialCpuQds != 0)
        {
          // internal buffers of SeqQDS are fixed during construction
          // but our input images can vary in size
          if ((m_SequentialCpuQds->GetWidth()  != width) ||
              (m_SequentialCpuQds->GetHeight() != height))
          {
            // will be recreated below, with the correct dimensions
            delete m_SequentialCpuQds;
            m_SequentialCpuQds = 0;
          }
        }

        // may not have been created before
        // or may have been deleted above in the size check
        if (m_SequentialCpuQds == 0)
          m_SequentialCpuQds = new SequentialCpuQds(width, height);

        m_SequentialCpuQds->Process(&leftIpl, &rightIpl);
        break;
      }

      default:
        throw std::logic_error("Method not implemented");
    }

    // FIXME: convert disparity image to point cloud
  }
  catch (const mitk::Exception& e)
  {
    throw std::runtime_error(std::string("Something went wrong with MITK bits: ") + e.what());
  }

}

} // end namespace
