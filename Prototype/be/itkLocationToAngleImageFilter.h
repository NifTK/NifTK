
#ifndef __itkLocationToAngleImageFilter_h
#define __itkLocationToAngleImageFilter_h



#include "itkVector.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkImageToImageFilter.h"

#include "itkVector.h"
#include "itkMatrix.h"


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
    typedef LocationToAngleImageFilter                                                  Self;
    typedef ImageToImageFilter< Image< TScalarType, NDimensions>,
                                Image< TScalarType, NDimensions>
                              >                                                         Superclass;
    typedef SmartPointer<Self>                                                          Pointer;
    typedef SmartPointer<const Self>                                                    ConstPointer;

    /** Standard typedefs. */
    typedef TScalarType																	InputPixelType;
    typedef Image< InputPixelType, NDimensions >                                        InputImageType;
    typedef typename InputImageType::IndexType                                          InputImageIndexType;
	typedef typename InputImageType::RegionType                                         InputImageRegionType;
    typedef float                                                                       OutputPixelType;
    typedef Image< OutputPixelType, NDimensions >                                       OutputImageType;
	typedef Vector< double, NDimensions >										        VectorType;
	typedef Point< double, NDimensions >                                                PointType;
	

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
