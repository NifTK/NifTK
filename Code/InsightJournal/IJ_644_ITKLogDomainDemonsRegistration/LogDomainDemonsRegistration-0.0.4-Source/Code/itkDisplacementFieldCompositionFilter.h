#ifndef __itkDisplacementFieldCompositionFilter_h
#define __itkDisplacementFieldCompositionFilter_h

#include <itkImageToImageFilter.h>
#include <itkWarpVectorImageFilter.h>
#include <itkAddImageFilter.h>


namespace itk
{
/** \class DisplacementFieldCompositionFilter
 * \brief Compute the composition of two displacement
 * fields.
 * 
 * Given two spatial transformations fl and fr represented
 * by the displacement fields dfl and dfr, this filter computes
 * the displacment field df that represents the spatial
 * transformation f = fl o fr.
 *
 * The convention used in this filter is to rely on the order
 * of the writing to distinguish the two field (the l letter stands
 * for left, the r letter stands for right). This is necessary
 * since the composition is not commutative.
 *
 * Note that the relationship between a transformation f and
 * the displacement field df that represents it is given by
 * f = Id + df so that a given point p gets transformed as
 * f(p) = p + df(p)
 *
 * The composition can then be expressed as
 * df = dfr + warp_{dfl}(dfr) where the warp operation results from
 * a WarpVectorImageFilter whose input is df2 and whose warp is df1.
 *
 * \author Florence Dru, INRIA and Tom Vercauteren, MKT
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DisplacementFieldCompositionFilter :
  public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DisplacementFieldCompositionFilter             Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>   Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** InputImage type. */
  typedef typename Superclass::InputImageType         DisplacementFieldType;
  typedef typename Superclass::InputImagePointer      DisplacementFieldPointer;
  typedef typename Superclass::InputImageConstPointer DisplacementFieldConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( DisplacementFieldImageFilter, ImageToImageFilter );

  /** Warper type. */
  typedef WarpVectorImageFilter<DisplacementFieldType,
      DisplacementFieldType,DisplacementFieldType>     VectorWarperType;
  typedef typename VectorWarperType::Pointer
    VectorWarperPointer;
  
  /** Set the warper. */
  itkSetObjectMacro( Warper, VectorWarperType );

  /** Get the warper (can be used to change the interpolator). */
  itkGetObjectMacro( Warper, VectorWarperType );


protected:
  DisplacementFieldCompositionFilter();
  ~DisplacementFieldCompositionFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** GenerateData() */
  void GenerateData();

  
  /** Adder type. */
  typedef AddImageFilter<DisplacementFieldType,DisplacementFieldType,
      DisplacementFieldType>                           AdderType;
  typedef typename AdderType::Pointer                  AdderPointer;

  /** Set the adder. */
  itkSetObjectMacro( Adder, AdderType );

  /** Get the adder. */
  itkGetObjectMacro( Adder, AdderType );

private:
  DisplacementFieldCompositionFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  VectorWarperPointer        m_Warper;
  AdderPointer               m_Adder;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDisplacementFieldCompositionFilter.txx"
#endif

#endif
