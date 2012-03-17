#ifndef itkUnitCubeCCCounter_h
#define itkUnitCubeCCCounter_h

#include <vector>

#include "itkUnitCubeNeighbors.h"

namespace itk
{

/** 
 * @brief Functor counting the number of connected components restricted in a 
 * unit cube. This class is used for topological number computation, and 
 * should be mostly useless in any other cases.
 */
template< typename TConnectivity, 
          typename TNeighborhoodConnectivity = 
            typename NeighborhoodConnectivity<TConnectivity>::Type >
class ITK_EXPORT UnitCubeCCCounter
  {
  public :
    UnitCubeCCCounter();
    ~UnitCubeCCCounter();
    
    unsigned int operator()() const;
    
    template<typename Iterator>
    void SetImage(Iterator imageBegin, Iterator imageEnd);
      
  private :
    template<typename C>
    static std::vector<bool> CreateConnectivityTest();
    
    static std::vector<bool> const m_NeighborhoodConnectivityTest;
    static std::vector<bool> const m_ConnectivityTest;
    
    char* m_Image;
    
    static UnitCubeNeighbors<TConnectivity, TNeighborhoodConnectivity > const m_UnitCubeNeighbors;
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUnitCubeCCCounter.txx"
#endif

#endif // itkUnitCubeCCCounter_h
