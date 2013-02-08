/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkEulerAffineTransformMatrixAndItsVariations_h
#define __itkEulerAffineTransformMatrixAndItsVariations_h

#include "itkEulerAffineTransform.h"
#include "itkImage.h"

#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_sparse_matrix.h>


namespace itk
{

  /** \class EulerAffineTransformMatrixAndItsVariations
   * \brief Class to apply the affine transformation matrix to a 3D image.
   */

  template <class TScalarType = double>
    class ITK_EXPORT EulerAffineTransformMatrixAndItsVariations : public Object
  {
    public:
      /** Standard class typedefs. */
      typedef EulerAffineTransformMatrixAndItsVariations    Self;
      typedef Object             														Superclass;
      typedef SmartPointer<Self>                            Pointer;
      typedef SmartPointer<const Self>                      ConstPointer;

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

      /** Run-time type information (and related methods). */
      itkTypeMacro(EulerAffineTransformMatrixAndItsVariations, Object);

      /** Some convenient typedefs. */
      typedef typename itk::Size<3>              																VolumeSizeType;

      typedef typename itk::EulerAffineTransform<double, 3, 3>        					EulerAffineTransformType;
      typedef typename EulerAffineTransformType::Pointer              					EulerAffineTransformPointerType;
      typedef typename EulerAffineTransformType::ParametersType 								EulerAffineTransformParametersType;

      /** Create a sparse matrix to store the affine transformation matrix coefficients */
      typedef vnl_sparse_matrix<TScalarType>           		SparseMatrixType;
      typedef vnl_matrix<TScalarType>           					FullMatrixType;
      typedef vnl_vector<TScalarType>                   	VectorType;

      /** Set the affine transformation */
      itkSetObjectMacro( AffineTransform, EulerAffineTransformType );
      /** Get the affine transformation */
      itkGetObjectMacro( AffineTransform, EulerAffineTransformType );

      /// Set the volume size
      void SetVolumeSize(const VolumeSizeType &r) {m_VolumeSize = r; m_FlagInitialised = false;}

      /// Calculate and return the affine transformation matrix
      void GetAffineTransformationSparseMatrix(SparseMatrixType &R, VolumeSizeType &inSize, EulerAffineTransformParametersType &parameters);

      /// Calculate and return the transpose of the affine transformation matrix
      void GetAffineTransformationSparseMatrixT(SparseMatrixType &R, SparseMatrixType &RTrans, VolumeSizeType &inSize);

      /// Calculate and return the multiplication of the affine transformation matrix and image vector
      void CalculteMatrixVectorMultiplication(SparseMatrixType &R, VectorType const& inputImageVector, VectorType &outputImageVector);

      /// Set the Finite Difference Method (FDM) difference value
      void SetFDMDifference(const double &diffVal) {m_FDMDiffValue = diffVal;}

      /// Calculate and return the gradient vector of the affine transformation matrix per each parameter (using FDM)
      void CalculteMatrixVectorGradient(VolumeSizeType &inSize, VectorType const& inputImageVector, VectorType &outputGradVector, EulerAffineTransformParametersType &parameters, unsigned int paraNum);

      /// Calculate and return the gradient vector of the affine transformation matrix per each parameter (overloaded using Jacobian)
      void CalculteMatrixVectorGradient(FullMatrixType &jacobianMatrix, VolumeSizeType &inSize, EulerAffineTransformParametersType &parameters);


    protected:
      EulerAffineTransformMatrixAndItsVariations();
      virtual ~EulerAffineTransformMatrixAndItsVariations(void) {};
      void PrintSelf(std::ostream& os, Indent indent) const;

      /// A pointer to the 3D volume size
      VolumeSizeType 																	m_VolumeSize;
      unsigned long int 															m_input3DImageTotalSize;

			/// FDM difference value
      double 																					m_FDMDiffValue;

      /// Flag indicating whether the object has been initialised
      bool 																						m_FlagInitialised;

      /** The affin transform core matrix and its inverse matrix */
      FullMatrixType																	m_affineCoreMatrix;
      // FullMatrixType																m_affineCoreMatrixInverse;

      /** The input and output coordinate vectors */
      VectorType 																			m_inputCoordinateVector;
      VectorType																			m_outputCoordinateVector;

      /** The affine transform */
      EulerAffineTransformType::Pointer 							m_AffineTransform;
			EulerAffineTransformType::JacobianType					m_JacobianArray;

      /** Create a sparse matrix to store the affine transformation matrix coefficients */
      // SparseMatrixType const* 												pSparseAffineTransformMatrix;

      /** Create a sparse matrix to store the transpose of the affine transformation matrix */
      // SparseMatrixType const* 												pSparseTransposeAffineTransformMatrix;


    private:
      EulerAffineTransformMatrixAndItsVariations(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

  };

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkEulerAffineTransformMatrixAndItsVariations.txx"
#endif

#endif
