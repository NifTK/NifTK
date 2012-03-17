#ifndef itkConnectivity_txx
#define itkConnectivity_txx

#include "itkConnectivity.h"

namespace itk
{

template<unsigned int VDim, unsigned int VCellDim>
Connectivity<VDim, VCellDim> const *
Connectivity<VDim, VCellDim>
::m_Instance = 0;

template<unsigned int VDim, unsigned int VCellDim>
Connectivity<VDim, VCellDim> const &
Connectivity<VDim, VCellDim>
::GetInstance()
  {
  if(m_Instance == 0)
    {
    m_Instance = new Connectivity<VDim, VCellDim>;
    }
    
  return (*m_Instance) ;
  }


template<unsigned int VDim, unsigned int VCellDim>
Connectivity<VDim, VCellDim>
::Connectivity()
: m_NeighborhoodSize(static_cast<int>(std::pow(3.0, static_cast<double>(VDim)))),
  m_NumberOfNeighbors( ComputeNumberOfNeighbors() ), 
  m_NeighborsPoints(new Point[m_NumberOfNeighbors] ),
  m_NeighborsOffsets(new Offset[m_NumberOfNeighbors] )
  {
  int currentNbNeighbors = 0;
  
  for(int i=0; i< m_NeighborhoodSize; ++i)
    {
    Point const p = OffsetToPoint(i);
        
    unsigned int const numberOfZeros = std::count(p.begin(), p.end(), 0);
        
    if( numberOfZeros!=VDim && numberOfZeros >= VCellDim)
      {
      *(m_NeighborsPoints+currentNbNeighbors) = p;
      *(m_NeighborsOffsets+currentNbNeighbors) = i;
      ++currentNbNeighbors;
      }
    }
  }


template<unsigned int VDim, unsigned int VCellDim>
Connectivity<VDim, VCellDim>
::~Connectivity()
  {
  delete[] m_NeighborsPoints;
  delete[] m_NeighborsOffsets;
  }


template<unsigned int VDim, unsigned int VCellDim>
bool
Connectivity<VDim, VCellDim>
::AreNeighbors(Point const & p1, Point const & p2) const
  {
  Point difference;
  for(unsigned int i=0; i<VDim; ++i)
    {
    difference[i] = p1[i] - p2[i];
    }
  
  Point* const iterator = 
    std::find(m_NeighborsPoints, m_NeighborsPoints+m_NumberOfNeighbors, 
              difference);
  return ( iterator != m_NeighborsPoints+m_NumberOfNeighbors );
  }


template<unsigned int VDim, unsigned int VCellDim>
bool
Connectivity<VDim, VCellDim>
::AreNeighbors(Offset const & o1, Point const & o2) const
  {
  /// @todo
  assert(false && "not implemented");
  return false;
  }
    
    
template<unsigned int VDim, unsigned int VCellDim>
bool 
Connectivity<VDim, VCellDim>
::IsInNeighborhood(Point const & p) const
  {
  Point* const iterator = 
    std::find(m_NeighborsPoints, m_NeighborsPoints+m_NumberOfNeighbors, p);
  return ( iterator != m_NeighborsPoints+m_NumberOfNeighbors );
  }


template<unsigned int VDim, unsigned int VCellDim>
bool 
Connectivity<VDim, VCellDim>
::IsInNeighborhood(Offset const & o) const
  {
  Offset* const iterator = 
    std::find(m_NeighborsOffsets, m_NeighborsOffsets+m_NumberOfNeighbors, o);
  return (iterator != m_NeighborsOffsets+m_NumberOfNeighbors);
  }


template<unsigned int VDim, unsigned int VCellDim>
typename Connectivity<VDim, VCellDim>::Point
Connectivity<VDim, VCellDim>
::OffsetToPoint(Offset const offset) const
  {
  Offset remainder = offset;
  Point p;
  
  for(unsigned int i=0; i<Dimension; ++i)
    {
    p[i] = remainder % 3;
    remainder -= p[i];
    remainder /= 3;
    --p[i];
    }
  
  return p;
  }


template<unsigned int VDim, unsigned int VCellDim>
typename Connectivity<VDim, VCellDim>::Offset
Connectivity<VDim, VCellDim>
::PointToOffset(Point const p) const
  {
  Offset offset=0;
  Offset factor=1;
  for(unsigned int i=0; i<Dimension; ++i)
    {
    offset += factor * (p[i]+1);
    factor *= 3;
    }
  
  return offset;
  }


int factorial(int n)
  {
  if(n<=1) return 1;
  else return n*factorial(n-1);
  }

template<unsigned int VDim, unsigned int VCellDim>
int 
Connectivity<VDim, VCellDim>
::ComputeNumberOfNeighbors()
  {
  int numberOfNeighbors = 0;
  for(unsigned int i = VCellDim; i <= VDim-1; ++i)
    {
    numberOfNeighbors += 
      factorial(VDim)/(factorial(VDim-i)*factorial(i)) * 1<<(VDim-i);
    }
  
  return numberOfNeighbors;
  }

}

#endif // itkConnectivity_txx
