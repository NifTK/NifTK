#ifndef itkTopologicalNumberImageFunction_h
#define itkTopologicalNumberImageFunction_h

#include <utility>

#include <itkImageFunction.h>

#include "itkBackgroundConnectivity.h"
#include "itkUnitCubeCCCounter.h"

namespace itk
{

/**
 * @brief Compute the topological numbers of an image at given index.
 *
 * Topological numbers characterize the topological properties of a point. They
 * are defined in an article by G. Bertrand and G. Malandain : "A new 
 * characterization of three-dimensional simple points"; Pattern Recognition 
 * Letters; 15:169--175; 1994.
 */
template<typename TImage, 
         typename TFGConnectivity, 
         typename TBGConnectivity = 
           typename BackgroundConnectivity<TFGConnectivity>::Type  >
class ITK_EXPORT TopologicalNumberImageFunction : 
  public itk::ImageFunction<TImage, std::pair<unsigned int, unsigned int> >
  {
  public :
    /**
     * @name Standard ITK declarations
     */
    //@{
    typedef TopologicalNumberImageFunction Self;
    typedef itk::ImageFunction<TImage, std::pair<unsigned int, unsigned int> > 
      Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<Self const> ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(TopologicalNumberImageFunction, ImageFunction);
    
    typedef typename Superclass::PointType PointType;
    typedef typename Superclass::ContinuousIndexType ContinuousIndexType;
    typedef typename Superclass::IndexType IndexType;
    //@}
    
    /**
     * @brief Initialize the functor so that the topological numbers are 
     * computed for both the foreground and the background.
     */
    TopologicalNumberImageFunction();
    
    /**
     * @name Evaluation functions
     *
     * These functions evaluate the topological number at the index.
     */
    //@{
    std::pair<unsigned int, unsigned int> 
      Evaluate(PointType const & point) const;
    
    std::pair<unsigned int, unsigned int> 
      EvaluateAtIndex(IndexType const & index) const;
    
    std::pair<unsigned int, unsigned int> 
      EvaluateAtContinuousIndex(ContinuousIndexType const & contIndex) const;
    //@}

    /**
     * @name Selectors for the computation of fore- and background topological 
     * numbers.
     *
     * These two members allow to selectively compute the topological
     * numbers for the background and the foreground. They are both set to true
     * during the construction of the object.
     */
    //@{
    itkGetConstMacro(ComputeForegroundTN, bool);
    itkSetMacro(ComputeForegroundTN, bool);
    itkGetConstMacro(ComputeBackgroundTN, bool);
    itkSetMacro(ComputeBackgroundTN, bool);
    //@}

  private :
    static UnitCubeCCCounter< TFGConnectivity > m_ForegroundUnitCubeCCCounter;
    static UnitCubeCCCounter< TBGConnectivity > m_BackgroundUnitCubeCCCounter;
    
    TopologicalNumberImageFunction(Self const &); // not implemented
    Self & operator=(Self const &); // not implemented
    
    bool m_ComputeForegroundTN;
    bool m_ComputeBackgroundTN;    
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTopologicalNumberImageFunction.txx"
#endif

#endif // itkTopologicalNumberImageFunction_h
