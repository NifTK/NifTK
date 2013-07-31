/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesData_h
#define mitkCoordinateAxesData_h

#include "niftkCoreExports.h"
#include <itkImageRegion.h>
#include <mitkBaseData.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class CoordinateAxesData
 * \brief Class to enable the data to represent coordinate axes to be
 * added as a mitk::BaseData to the mitk::DataStorage. This class
 * turns out to be very simple, as the BaseData contains the
 * mitk::Geometry3D, which contains all the information we need. So
 * this class is effectively a dummy class.
 */
class NIFTKCORE_EXPORT CoordinateAxesData : public BaseData {

public:

  typedef itk::ImageRegion<5>   RegionType;
  typedef RegionType::IndexType IndexType;
  typedef RegionType::SizeType  SizeType;

  mitkClassMacro(CoordinateAxesData, BaseData);
  itkNewMacro(Self);

  static const char* FILE_NAME;
  static const char* FILE_EXTENSION;
  static const char* FILE_EXTENSION_WITH_ASTERISK;
  static const char* FILE_DIALOG_PATTERN;
  static const char* FILE_DIALOG_NAME;

  void SetRequestedRegionToLargestPossibleRegion();
  bool RequestedRegionIsOutsideOfTheBufferedRegion();
  virtual bool VerifyRequestedRegion();
  void SetRequestedRegion(itk::DataObject *data);
  const RegionType& GetLargestPossibleRegion() const;
  virtual const RegionType& GetRequestedRegion() const;
  virtual void UpdateOutputInformation();

  void GetVtkMatrix(vtkMatrix4x4& matrixToWriteTo) const;
  void SetVtkMatrix(const vtkMatrix4x4& matrix);

protected:
  CoordinateAxesData();
  ~CoordinateAxesData();

private:
  CoordinateAxesData(const CoordinateAxesData&); // Purposefully not implemented.
  CoordinateAxesData& operator=(const CoordinateAxesData&); // Purposefully not implemented.

  mutable RegionType m_LargestPossibleRegion;
  RegionType m_RequestedRegion;
};

} // end namespace

#endif // MITKCOORDINATEAXESDATA_H
