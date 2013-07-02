/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_H_
#define ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_H_

#include <itkVector.h>
#include <itkMatrix.h>
#include <niftkITKWin32ExportHeader.h>

/*
 * \brief   Class for performing the pixel based operation
 *
 */
template<unsigned int Dimension = 3, class VectorComponentType = float>
class ITK_EXPORT itkDisplacementVectorCoordinateAdaptionPixelAccessor
{
  public:

	/*
	 * Some typedefs
	 */
	typedef itk::Vector<VectorComponentType, Dimension> InternalType;
	typedef itk::Vector<VectorComponentType, Dimension> ExternalType;


	typedef double MatrixElementType;
	typedef itk::Vector<MatrixElementType, Dimension + 1> InternalHomogenousVectorType;
	typedef itk::Matrix< MatrixElementType, Dimension + 1, Dimension + 1 > HomogenousMatrixType;

	/*
	 * \brief    Get the pixel value
	 * \details  The conversion of the pixel value is performed here.
	 */
	ExternalType Get( const InternalType & in ) const;


	/*
	 * \brief  Constructor
	 */
	itkDisplacementVectorCoordinateAdaptionPixelAccessor();


	/*
	 * \brief  Destructor
	 */
	virtual ~itkDisplacementVectorCoordinateAdaptionPixelAccessor();

	/*
	 * \brief  Set the homogenous matrix
	 */
	void SetHomogenousMatrix( const HomogenousMatrixType & hohogenousMatrixIn );

  private:
	HomogenousMatrixType m_matrix;  /// The internal transformation matrix.
};

#include "itkDisplacementVectorCoordinateAdaptionPixelAccessor.txx"

#endif /* ITKDISPLACEMENTVECTORCOORDINATEADAPTIONPIXELACCESSOR_H_ */
