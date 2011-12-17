#ifndef itkConnectivity_h
#define itkConnectivity_h

#include <itkMacro.h>

#include <vnl/vnl_vector_fixed.h>

namespace itk
{

/**
 * @brief Connectivity information.
 *
 * This class describes the k-neighbors of a point in a digital space of 
 * dimension n.
 * The definition used for this connectivity is the one using cell 
 * decomposition, as defined by Malandain in "On Topology in Multidimensional 
 * Discrete Spaces", ftp://ftp.inria.fr/INRIA/publication/RR/RR-2098.ps.gz
 *
 * The advantage of this definition instead of the 4- or 8-connectivity (in 2D)
 * or 6-, 18- and 26-connectivity (in 3D) is that it is consistent and 
 * extensible in n-dimensions.
 *
 * In 2D, 4- and 8-connectivity are respectively corresponding to 1- and
 * 0-connectivities. In 3D, 6-, 18- and 26-connectivity are respectively
 * corresponding to 2-, 1- and 0-connectivity.
 */
template<unsigned int VDim, unsigned int VCellDim>
class ITK_EXPORT Connectivity
  {
  public :
    /// @brief Type for a point in n-D.
    typedef vnl_vector_fixed<int, VDim> Point;
    
    /// @brief Offset in an array.
    typedef unsigned int Offset;
      
    /// @brief The dimension of the space.
    itkStaticConstMacro(Dimension, unsigned int, VDim);
      
    /// @brief The dimension of the cell.
    itkStaticConstMacro(CellDimension, unsigned int, VCellDim);
      
    unsigned int GetNeighborhoodSize() const
      {
      return m_NeighborhoodSize;
      }
      
    unsigned int GetNumberOfNeighbors() const
      {
      return m_NumberOfNeighbors;
      }
    
    /// @brief Accessor to the singleton.
    static Connectivity<VDim, VCellDim> const & GetInstance();
    
    Point const * const GetNeighborsPoints() const
      {
      return m_NeighborsPoints;
      }
      
    Point const * const GetNeighborsOffsets() const
      {
      return m_NeighborsOffsets;
      }
    
    /// @brief Test if two points are neighbors
    bool AreNeighbors(Point const & p1, Point const & p2) const;
    
    /// @brief Test if two points are neighbors
    bool AreNeighbors(Offset const & o1, Point const & o2) const;
    
    /// @brief Test if a point is a neighbor of 0
    bool IsInNeighborhood(Point const & p) const;
    
    /// @brief Test if a point is a neighbor of 0
    bool IsInNeighborhood(Offset const & o) const;
    
    /// @brief Convert an offset to a point, in a 3x3x3 cube
    Point OffsetToPoint(Offset const offset) const;
    
    /// @brief Convert a point to an offset, in a 3x3x3 cube
    Offset PointToOffset(Point const p) const;
    
  private :
     static Connectivity<VDim, VCellDim> const * m_Instance;
    
    /// @brief Size of the whole neighborhood.
    unsigned int const m_NeighborhoodSize;
    
    /// @brief Number of neighbors
    unsigned int const m_NumberOfNeighbors;
    
    /// @brief Neighbors as points.
    Point * const m_NeighborsPoints;
    
    /// @brief Neighbors as offsets.
    Offset * const m_NeighborsOffsets;

    Connectivity();
    ~Connectivity();
    
    // Purposedly not implemeted
    Connectivity(Connectivity<VDim, VCellDim> const & other);
    
    // Purposedly not implemeted
    Connectivity & operator=(Connectivity<VDim, VCellDim> const & other);
    
    static int ComputeNumberOfNeighbors();
  };

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkConnectivity.txx"
#endif

#endif // itkConnectivity_h
