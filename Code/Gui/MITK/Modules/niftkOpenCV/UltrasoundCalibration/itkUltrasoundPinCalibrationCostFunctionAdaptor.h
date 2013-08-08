/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUltrasoundPinCalibrationCostFunctionAdaptor_h
#define itkUltrasoundPinCalibrationCostFunctionAdaptor_h

namespace itk {

/**
 * \class UltrasoundPinCalibrationCostFunctionAdaptor
 * \brief Multi-valued cost function adaptor, to plug into Levenberg-Marquardt.
 */
class UltrasoundPinCalibrationCostFunctionAdaptor
{
public:
    UltrasoundPinCalibrationCostFunctionAdaptor();
};

}

#endif
