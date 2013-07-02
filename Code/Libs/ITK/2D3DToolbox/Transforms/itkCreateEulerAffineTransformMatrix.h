/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCreateEulerAffineTransformMatrix_h
#define __itkCreateEulerAffineTransformMatrix_h

#include "itkCreateEulerAffineTransformMatrixBaseClass.h"
#include <itkEulerAffineTransform.h>
#include <itk_hash_map.h>

#include <vnl/vnl_math.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_sparse_matrix.h>


namespace itk
{
  
/** \class CreateEulerAffineTransformMatrix
 * \brief Class to apply the affine transformation matrix to a 3D image.
 */

template <class IntensityType = float>
class ITK_EXPORT CreateEulerAffineTransformMatrix :
    public CreateEulerAffineTransformMatrixBaseClass<Image< IntensityType, 3>,  // Input image
					Image< IntensityType, 3> > // Output image
{
public:
  /** Standard class typedefs. */
  typedef CreateEulerAffineTransformMatrix Self;
  typedef SmartPointer<Self>               Pointer;
  typedef SmartPointer<const Self>         ConstPointer;
  typedef CreateEulerAffineTransformMatrixBaseClass<Image< IntensityType, 3>, Image< IntensityType, 3> >  	Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CreateEulerAffineTransformMatrix, CreateEulerAffineTransformMatrixBaseClass);

  /** Some convenient typedefs. */
  typedef Image<IntensityType, 3>               	InputImageType;
  typedef typename InputImageType::Pointer     		InputImagePointer;
  typedef typename InputImageType::ConstPointer 	InputImageConstPointer;
  typedef typename InputImageType::RegionType  		InputImageRegionType;
  typedef typename InputImageType::SizeType    		InputImageSizeType;
  typedef typename InputImageType::SpacingType 		InputImageSpacingType;
  typedef typename InputImageType::PointType   		InputImagePointType;
  typedef typename InputImageType::PixelType   		InputImagePixelType;
  typedef typename InputImageType::IndexType   		InputImageIndexType;


  typedef Image<IntensityType, 3>               	OutputImageType;
  typedef typename OutputImageType::Pointer     	OutputImagePointer;
	typedef typename OutputImageType::ConstPointer 	OutputImageConstPointer;
  typedef typename OutputImageType::RegionType  	OutputImageRegionType;
  typedef typename OutputImageType::SizeType    	OutputImageSizeType;
  typedef typename OutputImageType::SpacingType 	OutputImageSpacingType;
  typedef typename OutputImageType::PointType   	OutputImagePointType;
  typedef typename OutputImageType::PixelType   	OutputImagePixelType;
  typedef typename OutputImageType::IndexType   	OutputImageIndexType;

  typedef EulerAffineTransform<double, 3, 3> EulerAffineTransformType;

  /** Create a sparse matrix to store the affine transformation matrix coefficients */
  typedef vnl_sparse_matrix<double>           	SparseMatrixType;
	typedef vnl_matrix<double>           					FullMatrixType;
  typedef vnl_vector<double>                    VectorType;

  /** Type of the map used to store the affine transformation matrix per thread */
  typedef itk::hash_map<int, VectorType>  																VectorMapType;
  typedef typename itk::hash_map<int, VectorType>::iterator 							VectorMapIterator;
  typedef typename itk::hash_map<int, VectorType>::const_iterator 				VectorMapConstIterator;

  /** ImageDimension enumeration */
  itkStaticConstMacro(InputImageDimension, unsigned int, 3);
  itkStaticConstMacro(OutputImageDimension, unsigned int, 3);
  /** Set the affine transformation */
  itkSetObjectMacro( AffineTransform, EulerAffineTransformType );
  /** Get the affine transformation */
  itkGetObjectMacro( AffineTransform, EulerAffineTransformType );

  /** CreateEulerAffineTransformMatrix produces a 3D ouput image which is a same
   * resolution and with a different pixel spacing than its 3D input
   * image (obviously).  As such, CreateEulerAffineTransformMatrix needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model. The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  virtual void GenerateOutputInformation(void);

  /** Rather than calculate the input requested region for a
   * particular projection (which might take longer than the actual
   * projection), we simply set the input requested region to the
   * entire 3D input image region. Therefore needs to provide an implementation
   * for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.  \sa
   * ProcessObject::GenerateInputRequestedRegion() */
  virtual void GenerateInputRequestedRegion(void);
  virtual void EnlargeOutputRequestedRegion(DataObject *output); 

  /** Set the size in pixels of the output projected image. */
  void SetTransformedImageSize(OutputImageSizeType &outImageSize) {m_OutputImageSize = outImageSize;};
  /** Set the resolution in mm of the output projected image. */
  void SetTransformedImageSpacing(OutputImageSpacingType &outImageSpacing) {
    m_OutputImageSpacing = outImageSpacing;
    this->GetOutput()->SetSpacing(m_OutputImageSpacing);
  };
  /** Set the origin of the output projected image. */
  void SetTransformedImageOrigin(OutputImagePointType &outImageOrigin) {
    m_OutputImageOrigin = outImageOrigin;
    this->GetOutput()->SetOrigin(m_OutputImageOrigin);
  };

  /** For debugging purposes, set single threaded execution */
  void SetSingleThreadedExecution(void) {m_FlagMultiThreadedExecution = false;}

protected:
  CreateEulerAffineTransformMatrix();
  virtual ~CreateEulerAffineTransformMatrix(void) {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** If an imaging filter needs to perform processing after the buffer
   * has been allocated but before threads are spawned, the filter can
   * can provide an implementation for BeforeThreadedGenerateData(). The
   * execution flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void BeforeThreadedGenerateData(void);
  
  /** If an imaging filter needs to perform processing after all
   * processing threads have completed, the filter can can provide an
   * implementation for AfterThreadedGenerateData(). The execution
   * flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void AfterThreadedGenerateData(void);
  
  /** Single threaded execution, for debugging purposes ( call
  SetSingleThreadedExecution() ) */
  void GenerateData();

  /** CreateEulerAffineTransformMatrix can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData()
   * routine which is called for each processing thread. The output
   * image data is allocated automatically by the superclass prior to
   * calling ThreadedGenerateData().  ThreadedGenerateData can only
   * write to the portion of the output image specified by the
   * parameter "outputRegionForThread"
   *
   * \sa CreateEulerAffineTransformMatrixBaseClass::ThreadedGenerateData(),
   *     CreateEulerAffineTransformMatrixBaseClass::GenerateData() */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                            int threadId );

  /** The size of the output projected image */
  OutputImageSizeType m_OutputImageSize;
  /** The resolution of the output projected image */
  OutputImageSpacingType m_OutputImageSpacing;
  /** The origin of the output projected image */
  OutputImagePointType m_OutputImageOrigin;

  /** Flag to turn multithreading on or off */
  bool m_FlagMultiThreadedExecution;

  /** The affin transform core matrix and its inverse matrix */
	FullMatrixType																	m_affineCoreMatrix;
	FullMatrixType																	m_affineCoreMatrixInverse;

	/** The input and output coordinate vectors */
	VectorType 																			m_inputCoordinateVector;
	VectorType																			m_outputCoordinateVector;

  /** The affine transform */
  EulerAffineTransformType::Pointer 							m_AffineTransform;

  /** Create a sparse matrix to store the affine transformation matrix coefficients */
  SparseMatrixType 																m_sparseAffineTransformMatrix;

	/** Create a sparse matrix to store the transpose of the affine transformation matrix */
	SparseMatrixType 																m_sparseTransposeAffineTransformMatrix;

private:
  CreateEulerAffineTransformMatrix(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCreateEulerAffineTransformMatrix.txx"
#endif

#endif
