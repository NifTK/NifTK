/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTriangulate2DPointPairsTo3D.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <FileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
Triangulate2DPointPairsTo3D::Triangulate2DPointPairsTo3D()
{

}


//-----------------------------------------------------------------------------
Triangulate2DPointPairsTo3D::~Triangulate2DPointPairsTo3D()
{

}



//-----------------------------------------------------------------------------
bool Triangulate2DPointPairsTo3D::Triangulate(const std::string& input2DPointPairsFileName,
                                              const std::string& intrinsicLeftFileName,
                                              const std::string& intrinsicRightFileName,
                                              const std::string& rightToLeftRotationFileName,
                                              const std::string& rightToLeftTranslationFileName,
                                              const std::string& goldStandard3DPointsFileName
                                             )
{
  bool isSuccessful = false;
  return isSuccessful;
}

} // end namespace
