/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkStereoImageToModelSSD_h
#define mitkStereoImageToModelSSD_h

#include "mitkStereoImageToModelMetric.h"

namespace mitk
{

/**
 * \class StereoImageToModelSSD
 * \brief Computes similarity between a model (currently VTK), and two stereo images using SSD of intensity values.
 */

class StereoImageToModelSSD : public StereoImageToModelMetric
{
public:

  /** Standard class typedefs. */
  typedef StereoImageToModelSSD         Self;
  typedef StereoImageToModelMetric      Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(StereoImageToModelSSD, StereoImageToModelMetric);
  itkNewMacro(StereoImageToModelSSD);

  /**  Type of the measure. */
  typedef Superclass::MeasureType        MeasureType;

  /**  Type of the derivative. */
  typedef Superclass::DerivativeType     DerivativeType;

  /**  Type of the parameters. */
  typedef Superclass::ParametersType     ParametersType;

protected:

  StereoImageToModelSSD();
  virtual ~StereoImageToModelSSD();

  virtual MeasureType CalculateCost(
      const CvMat& transformed3DPoints,
      const CvMat& transformed3DNormals,
      const CvMat& weights,
      const CvMat& transformed2DPointsLeft,
      const CvMat& transformed2DPointsRight,
      const ParametersType &parameters
      ) const ;

private:
  StereoImageToModelSSD(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class

} // end namespace mitk

#endif // MITKSTEREOIMAGETOMODELSSD_H

