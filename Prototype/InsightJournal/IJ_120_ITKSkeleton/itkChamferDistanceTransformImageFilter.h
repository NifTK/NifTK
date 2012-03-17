#ifndef itkChamferDistanceTransformImageFilter_h
#define itkChamferDistanceTransformImageFilter_h

#include <iosfwd>
#include <vector>

#include <itkImageToImageFilter.h>

namespace itk
{

/**
 * @brief Compute the chamfer distance on an image.
 *
 * This is the two pass algorithm of Borgefors in "On digital distance 
 * transforms in three dimensions", Computer Vision and Image Understanding
 * 64(3), pp. 368--376, 1996.
 */
template<typename InputImage, typename OutputImage>
class ITK_EXPORT ChamferDistanceTransformImageFilter : 
  public ImageToImageFilter<InputImage, OutputImage>
  {
  public :
    /**
     * @name Standard ITK declarations
     */
    //@{
    typedef ChamferDistanceTransformImageFilter Self;
    typedef ImageToImageFilter<InputImage, OutputImage> Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<Self const> ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(ChamferDistanceTransformImageFilter, ImageToImageFilter);

    //@}

    /**
     * @name Standard filter typedefs.
     */
    //@{
    typedef InputImage InputImageType;
    typedef OutputImage OutputImageType;
    //@}

    /**
     * @brief Initializes the filter.
     * 
     * Weights are set to 1, and distanceFromObject is set to true.
     */
    ChamferDistanceTransformImageFilter();

    itkGetConstMacro(DistanceFromObject, bool);
    itkSetMacro(DistanceFromObject, bool);

    /**
     * @brief Assign the weights used in the distance transform.
     *
     * @pre The sequence (begin, end) must not be larger than the dimension
     * of the output image.
     */
    template<typename Iterator>
    void SetWeights(Iterator begin, Iterator end);

    /**
     * @brief Return the weights used in the distance transform.
     */
    std::vector<typename OutputImage::PixelType> GetWeights() const;

  protected :
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    void GenerateData();    

  private :
    typename OutputImage::PixelType m_Weights[OutputImage::ImageDimension];
    
    ChamferDistanceTransformImageFilter(Self const &); // not implemented
    Self & operator=(Self const &); // not implemented

    /**
     * @brief Select if the distance is computed from the object (in the 
     * background) or from the background (in the object).
     *
     * It is initialized to true, so by default the distance is computed
     * from the object, in the background.
     */
    bool m_DistanceFromObject;
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkChamferDistanceTransformImageFilter.txx"
#endif

#endif // itkChamferDistanceTransformImageFilter_h
