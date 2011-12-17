#ifndef itkBackgroundConnectivity_h
#define itkBackgroundConnectivity_h

#include "itkConnectivity.h"

namespace itk
{

/**
 * @brief Helper to determine the background connectivity
 * @param FGConnectivity : the image connectivity
 */
template<typename TConnectivity>
struct ITK_EXPORT BackgroundConnectivity
  {
  typedef Connectivity<TConnectivity::Dimension, 
                       TConnectivity::Dimension-1 > Type;
  };

template<>
struct BackgroundConnectivity< Connectivity<2,1> >
  {
  typedef Connectivity<2, 0> Type;
  };

template<>
struct BackgroundConnectivity< Connectivity<3,2> >
  {
  typedef Connectivity<3, 0> Type;
  };

}

#endif // itkBackgroundConnectivity_h
