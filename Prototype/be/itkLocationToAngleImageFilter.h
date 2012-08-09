
#ifndef __itkLocationToAngleImageFilter_h
#define __itkLocationToAngleImageFilter_h



#include "itkVector.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkImageToImageFilter.h"

#include "itkVector.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_inverse.h"
#include "vnl/vnl_cross.h"


//#define _DBG

namespace itk {
	/**
	* \class LocationToAngleImageFilter
	* \brief This class considers the angle of a pixel with respect to a plane (three points) and a line through these planes.
	* 
	*/
	template < typename TScalarType, unsigned int NDimensions = 3>
	class ITK_EXPORT LocationToAngleImageFilter : 
		public ImageToImageFilter<
		Image< TScalarType, NDimensions>, // Input image
		Image< TScalarType, NDimensions>  // Output image
		>
	{
	public:

		/** Standard "Self" typedef. */
		typedef LocationToAngleImageFilter                                     Self;
		typedef ImageToImageFilter< Image< TScalarType, NDimensions>,
			Image< TScalarType, NDimensions>
		>                                                                      Superclass;
		typedef SmartPointer<Self>                                             Pointer;
		typedef SmartPointer<const Self>                                       ConstPointer;

		/** Standard typedefs. */
		typedef TScalarType                                                    InputPixelType;
		typedef Image< InputPixelType, NDimensions >                           InputImageType;
		typedef typename InputImageType::IndexType                             InputImageIndexType;
		typedef typename InputImageType::RegionType                            InputImageRegionType;
		typedef float                                                          OutputPixelType;
		typedef Image< OutputPixelType, NDimensions >                          OutputImageType;
		typedef Vector< double, 3 >                                            Vector3Type;
		typedef Vector< double, 2 >                                            Vector2Type;
		typedef Matrix< double, 3, 3 >                                         Matrix3x3Type;
		typedef Matrix< double, 3, 2 >                                         Matrix3x2Type;
		typedef Matrix< double, 3, 1 >                                         Matrix3x1Type;
		typedef Matrix< double, 2, 2 >                                         Matrix2x2Type;
		typedef Matrix< double, 2, 3 >                                         Matrix2x3Type;
		typedef Matrix< double, 1, 3 >                                         Matrix1x3Type;
		typedef Matrix< double, 1, 1 >                                         Matrix1x1Type;
		typedef Point < double, 3 >                                            PointType;

		// vnl typedefs
		typedef vnl_matrix< double >                                           VNLMatrixType;
		typedef vnl_vector< double >                                           VNLVectorType;

		/** Method for creation through the object factory. */
		itkNewMacro( Self );

		/** Run-time type information (and related methods). */
		itkTypeMacro(LocationToAngleImageFilter, ImageToImageFilter);

		/** Get the number of dimensions we are working in. */
		itkStaticConstMacro(Dimension, unsigned int, NDimensions);

		itkSetMacro( BodyAxisPoint1, PointType )
			itkSetMacro( BodyAxisPoint2, PointType )
			itkSetMacro( UpPoint,        PointType )

	protected:
		LocationToAngleImageFilter();
		~LocationToAngleImageFilter() {};
		void PrintSelf(std::ostream& os, Indent indent) const;

		// Check before we start.
		virtual void BeforeThreadedGenerateData();

		// The main method to implement in derived classes, note, its threaded.
		virtual void ThreadedGenerateData( const InputImageRegionType &outputRegionForThread, int);

	private:

		PointType m_BodyAxisPoint1;
		PointType m_BodyAxisPoint2;
		PointType m_UpPoint;

		// Vectors related to the actual angle calculations
		// All these take place in 
		VNLMatrixType m_mA1;  // Vector from first body-axis point to upwards pointing vector
		VNLMatrixType m_mA2;  // Vector from the first body-axis to the second
		double        m_dA2NormSquared;
		const double  m_dEpsiplon;

		VNLMatrixType m_mA;        // matrix A (first column holds vector m_vA1, second column holds m_vA2)
		VNLMatrixType m_mAT;       // transpose of A (A-transpose)
		VNLMatrixType m_mATAinvAT; // 
		VNLMatrixType m_mC1 ;      // Vector orthogonal to A2 and A1 (basically cross product A2xA1)
		
		/**
		* Prohibited copy and assignment. 
		*/
		LocationToAngleImageFilter(const Self&); 
		void operator=(const Self&); 

	}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLocationToAngleImageFilter.txx"
#endif

#endif
