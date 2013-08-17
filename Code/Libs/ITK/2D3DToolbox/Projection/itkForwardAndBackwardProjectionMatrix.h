/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkForwardAndBackwardProjectionMatrix_h
#define itkForwardAndBackwardProjectionMatrix_h

#include "itkRay.h"
#include <itkImage.h>

#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_sparse_matrix.h>

#include "itkProjectionGeometry.h"
#include <itkEulerAffineTransform.h>
#include <itkPerspectiveProjectionTransform.h>

namespace itk
{

  /** \class ForwardAndBackwardProjectionMatrix
   * \brief Class to apply the affine transformation matrix to a 3D image.
   */

  template <class TScalarType = double, class IntensityType = float>
    class ITK_EXPORT ForwardAndBackwardProjectionMatrix : public Object
  {
    public:
      /** Standard class typedefs. */
      typedef ForwardAndBackwardProjectionMatrix    				Self;
      typedef Object             														Superclass;
      typedef SmartPointer<Self>                            Pointer;
      typedef SmartPointer<const Self>                      ConstPointer;

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

      /** Run-time type information (and related methods). */
      itkTypeMacro(ForwardAndBackwardProjectionMatrix, Object);

      /** Some convenient typedefs. */
      typedef typename itk::Size<3>              											VolumeSizeType;

      typedef Image<IntensityType, 3>               									InputImageType;
      typedef typename InputImageType::Pointer      									InputImagePointer;
      typedef typename InputImageType::ConstPointer 									InputImageConstPointer;
      typedef typename InputImageType::RegionType   									InputImageRegionType;
      typedef typename InputImageType::PixelType    									InputImagePixelType;
      typedef typename InputImageType::SizeType     									InputImageSizeType;
      typedef typename InputImageType::SpacingType  									InputImageSpacingType;
      typedef typename InputImageType::PointType   										InputImagePointType;
      typedef typename InputImageType::IndexType   										InputImageIndexType;

      typedef Image<IntensityType, 2>               									OutputImageType;
      typedef typename OutputImageType::Pointer     									OutputImagePointer;
      typedef typename OutputImageType::ConstPointer 									OutputImageConstPointer;
      typedef typename OutputImageType::RegionType  									OutputImageRegionType;
      typedef typename OutputImageType::PixelType   									OutputImagePixelType;
      typedef typename OutputImageType::SizeType    									OutputImageSizeType;
      typedef typename OutputImageType::SpacingType 									OutputImageSpacingType;
      typedef typename OutputImageType::PointType   									OutputImagePointType;
      typedef typename OutputImageType::IndexType   									OutputImageIndexType;

      typedef itk::ProjectionGeometry<IntensityType> 									ProjectionGeometryType;
      typedef typename ProjectionGeometryType::Pointer 								ProjectionGeometryPointer;

      typedef itk::EulerAffineTransform<double, 3, 3> 								EulerAffineTransformType;
      typedef typename EulerAffineTransformType::Pointer 							EulerAffineTransformPointer;

      typedef itk::PerspectiveProjectionTransform<double> 						PerspectiveProjectionTransformType;
      typedef typename PerspectiveProjectionTransformType::Pointer 		PerspectiveProjectionTransformPointer;

      /// Get/Set the projection geometry
      itkSetObjectMacro( ProjectionGeometry, ProjectionGeometryType );
      itkGetObjectMacro( ProjectionGeometry, ProjectionGeometryType );
      /// Set the affine transformation
      itkSetObjectMacro( AffineTransform, EulerAffineTransformType );
      /// Get the affine transformation
      itkGetObjectMacro( AffineTransform, EulerAffineTransformType );
      /// Set the perspective transformation
      itkSetObjectMacro( PerspectiveTransform, PerspectiveProjectionTransformType );
      /// Get the perspective transformation
      itkGetObjectMacro( PerspectiveTransform, PerspectiveProjectionTransformType );


      /// Set the size in pixels of the output projected image.
      void SetProjectedImageSize(OutputImageSizeType &outImageSize) {m_OutputImageSize = outImageSize;};
      /// Set the resolution in mm of the output projected image.
      void SetProjectedImageSpacing(OutputImageSpacingType &outImageSpacing) {
        m_OutputImageSpacing = outImageSpacing;
        this->GetOutput()->SetSpacing(m_OutputImageSpacing);
      };
      /// Set the origin of the output projected image.
      void SetProjectedImageOrigin(OutputImagePointType &outImageOrigin) {
        m_OutputImageOrigin = outImageOrigin;
        this->GetOutput()->SetOrigin(m_OutputImageOrigin);
      };


      /** Create a sparse matrix to store the affine transformation matrix coefficients */
      typedef vnl_sparse_matrix<TScalarType>           					SparseMatrixType;
      typedef vnl_matrix<TScalarType>           								FullMatrixType;
      typedef vnl_vector<TScalarType>                   				VectorType;

      /// Set the volume size
      void SetVolumeSize(const VolumeSizeType &r) {m_VolumeSize = r; m_FlagInitialised = false;}

      /// Calculate and return the affine transformation matrix
      void GetForwardProjectionSparseMatrix(SparseMatrixType &R, InputImageConstPointer inImage, OutputImagePointer outImage,
          VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum);

      /// Calculate and return the affine transformation matrix (Overloaded using non-const input image pointer)
      void GetForwardProjectionSparseMatrix(SparseMatrixType &R, InputImagePointer inImage, OutputImagePointer outImage,
          VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum);

      /// Calculate and return the transpose of the affine transformation matrix
      void GetBackwardProjectionSparseMatrix(SparseMatrixType &R, SparseMatrixType &RTrans, 
          VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum);

      /// Calculate and return the multiplication of the affine transformation matrix and image vector
      void CalculteMatrixVectorMultiplication(SparseMatrixType &R, VectorType const &inputImageVector, VectorType &outputImageVector);


    protected:
      ForwardAndBackwardProjectionMatrix();
      virtual ~ForwardAndBackwardProjectionMatrix(void) {};
      void PrintSelf(std::ostream& os, Indent indent) const;

      /// A pointer to the 3D volume size
      VolumeSizeType 				m_VolumeSize;
      unsigned long int			m_input3DImageTotalSize;
      unsigned long int			m_output2DImageTotalSize;

      /// Flag indicating whether the object has been initialised
      bool m_FlagInitialised;

      /// The size of the output projected image
      OutputImageSizeType 		m_OutputImageSize;
      /// The resolution of the output projected image
      OutputImageSpacingType 	m_OutputImageSpacing;
      /// The origin of the output projected image
      OutputImagePointType	 	m_OutputImageOrigin;

      /// The threshold above which voxels along the ray path are integrated.
      double m_Threshold;

      /// The specific projection geometry to be used
      ProjectionGeometryPointer m_ProjectionGeometry;

      /// The affine transform
      EulerAffineTransformType::Pointer m_AffineTransform; 

      /// The perspective transform
      PerspectiveProjectionTransformType::Pointer m_PerspectiveTransform;

      /**
       *   The ray is traversed by stepping in the axial direction
       *   that enables the greatest number of planes in the volume to be
       *   intercepted.
       */
      typedef enum {
        UNDEFINED_DIRECTION=0,        //!< Undefined
        TRANSVERSE_IN_X,              //!< x
        TRANSVERSE_IN_Y,              //!< y
        TRANSVERSE_IN_Z,              //!< z
        LAST_DIRECTION
      } TraversalDirection;

      /** Create a sparse matrix to store the affine transformation matrix coefficients */
      // SparseMatrixType const* 												pSparseForwardProjMatrix;

      /** Create a sparse matrix to store the transpose of the affine transformation matrix */
      // SparseMatrixType const* 												pSparseBackwardProjMatrix;


    private:
      ForwardAndBackwardProjectionMatrix(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

  };

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForwardAndBackwardProjectionMatrix.txx"
#endif

#endif
