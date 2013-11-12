/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTriangulate2DPointPairsTo3D_h
#define mitkTriangulate2DPointPairsTo3D_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class Triangulate2DPointPairsTo3D
 * \brief Takes an input file containing 4 numbers on each line corresponding
 * to the x and y image coordinates for the left and then right image of a stereo
 * video pair, and all the calibration data to enable a reconstruction of 3D points.
 *
 * Currently the image that you are dealing with and hence the 2D pixel coordinates are assumed to be distortion corrected.
 */
class NIFTKOPENCV_EXPORT Triangulate2DPointPairsTo3D : public itk::Object
{

public:

  mitkClassMacro(Triangulate2DPointPairsTo3D, itk::Object);
  itkNewMacro(Triangulate2DPointPairsTo3D);

  bool Triangulate(const std::string& input2DPointPairsFileName,
      const std::string& intrinsicLeftFileName,
      const std::string& intrinsicRightFileName,
      const std::string& rightToLeftExtrinsics,
      const std::string& outputFileName
      );

protected:

  Triangulate2DPointPairsTo3D();
  virtual ~Triangulate2DPointPairsTo3D();

  Triangulate2DPointPairsTo3D(const Triangulate2DPointPairsTo3D&); // Purposefully not implemented.
  Triangulate2DPointPairsTo3D& operator=(const Triangulate2DPointPairsTo3D&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
