/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

//________________________________________________
// ALGORITHM AND SOURCE WAS ADOPTED FROM:
/**
*  @file    MarchingCubes.h
*  @author  Thomas Lewiner <tomlew@mat.puc-rio.br>
*  @author  Math Dept, PUC-Rio
*  @version 2008.1
*  @date    07/03/2008
*
*  @brief   MarchingCubes CMC33 Algorithm
* 
*  Updated to include the updates and corrections from:
*    Custodio, Lis, et al. "Practical considerations on Marching Cubes 33 topological correctness." Computers & Graphics (2013).
*    http://liscustodio.github.io/C_MC33/
*/
//________________________________________________


#ifndef mitkNifTKCMC33_h
#define mitkNifTKCMC33_h

#include "niftkCoreExports.h"
#include "mitkBasicVertex.h"
#include "mitkBasicTriangle.h"

namespace mitk {

  //_____________________________________________________________________________
  // types
  /** unsigned char alias */
  typedef unsigned char uchar;
  /** signed char alias */
  typedef   signed char schar;
  /** isovalue alias */ 
  typedef        float real ; 



  /**
  * \class MeshData
  * \brief Container class for passing around data structures containing a mesh
  */
  struct MeshData
  {
    std::vector<BasicVertex>            m_Vertices  ;  /**< vertex   buffer */
    std::vector<BasicTriangle>          m_Triangles ;  /**< triangle buffer */
    std::vector< std::vector<size_t> >  m_VertexToTriangleIndices; // lookup table vertex-->triangle
    std::vector< std::vector<size_t> >  m_VertexToVertexIndices;   // vertex adjacency list
  };



  //_____________________________________________________________________________
  /** Marching Cubes algorithm wrapper */
  /** \class CMC33
  * \brief Marching Cubes - CMC33 algorithm.
  */
  class NIFTKCORE_EXPORT CMC33
    //-----------------------------------------------------------------------------
  {
    // Constructors
  public :
    /**
    * Main and default constructor
    * \brief constructor
    * \param size_x width  of the grid
    * \param size_y depth  of the grid
    * \param size_z height of the grid
    */
    CMC33 ( const int size_x = -1, const int size_y = -1, const int size_z = -1 );
    /** Destructor */
    ~CMC33();

    //-----------------------------------------------------------------------------
    // Accessors
  public :
    /** accesses the number of vertices of the generated mesh */
    inline const int nverts() const { return _nverts; }
    /** accesses the number of triangles of the generated mesh */
    inline const int ntrigs() const { return _ntrigs; }

    /**  accesses the width  of the grid */
    inline const int size_x() const { return _size_x; }
    /**  accesses the depth  of the grid */
    inline const int size_y() const { return _size_y; }
    /**  accesses the height of the grid */
    inline const int size_z() const { return _size_z; }

    /**
    * changes the size of the grid
    * \param size_x width  of the grid
    * \param size_y depth  of the grid
    * \param size_z height of the grid
    */
    //inline void set_resolution( const int size_x, const int size_y, const int size_z ) { _size_x = size_x;  _size_y = size_y;  _size_z = size_z; }
    /**
    * selects wether the algorithm will use the enhanced topologically controlled lookup table or the original MarchingCubes
    * \param originalMC true for the original Marching Cubes
    */
    inline void set_method    ( const bool originalMC = false ) { _originalMC = originalMC; }
    
    /**
    * selects to use data from another class
    * \param data is the pointer to the external data, allocated as a size_x*size_y*size_z vector running in x first
    */
    inline void set_input_data  ( real *data ) { if( !_ext_data ) delete [] _data;  _ext_data = data != NULL;  if( _ext_data ) _data = data; }
    
    /** Set pointer to the data structure that will be used internally to store the mesh data*/
    inline void  set_output_data  (mitk::MeshData * meshData) { m_MeshDataExt = meshData; } 

    /** turns normal computing on / off */
    inline void enable_normal_computing(bool value) { _computeNormals = value; }

    // Data initialization
    /** inits temporary structures (must set sizes before call) : the grid and the vertex index per cube */
    void init_temps ();
    /** inits all structures (must set sizes before call) : the temporary structures and the mesh buffers */
    void init_all   ();
    /** clears temporary structures : the grid and the main */
    void clean_temps();
    /** clears all structures : the temporary structures and the mesh buffers */
    void clean_all  ();
    /** restart with a new mesh: erases all vertices and faces */
    void restart();

    //-----------------------------------------------------------------------------
    // Algorithm
  public :
    /**
    * Main algorithm : must be called after init_all
    * \param iso isovalue
    */
    void run( real iso = (real)0.0 ); 

  protected :
    /** tesselates one cube */
    void process_cube ()            ;
    /** tests if the components of the tesselation of the cube should be connected by the interior of an ambiguous face */
    bool test_face    ( schar face );
    /** tests if the components of the tesselation of the cube should be connected through the interior of the cube */
    bool test_interior( schar s )   ;
    /** IMPROVED -  tests if the components of the tesselation of the cube should be connected through the interior of the cube */
    bool modified_test_interior(schar s);
    /** tests the ambigous cases */
    int interior_ambiguity(int amb_face, int s);
    /** verify that the ambigous cases were treated right */
    int interior_ambiguity_verification(int edge);

    bool interior_test_case13();
    bool interior_test_case13_2(float isovalue);

    // Data access
    /**
    * accesses a specific cube of the grid
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline const real get_data  ( const int i, const int j, const int k ) const { return _data[ i + j*_size_x + k*_size_x*_size_y]; } 

    //-----------------------------------------------------------------------------
    // Operations
  protected :
    /**
    * computes almost all the vertices of the mesh by interpolation along the cubes edges
    * \param iso isovalue
    */
    void compute_intersection_points( real iso ); 

    /**
    * routine to add a triangle to the mesh
    * \param trig the code for the triangle as a sequence of edges index
    * \param n    the number of triangles to produce
    * \param v12  the index of the interior vertex to use, if necessary
    */
    void add_triangle ( const char* trig, char n, int v12 = -1 );

    /** tests and eventually doubles the vertex buffer capacity for a new vertex insertion */
    void test_vertex_addition();
    /** adds a vertex on the current horizontal edge */
    int add_x_vertex();
    /** adds a vertex on the current longitudinal edge */
    int add_y_vertex();
    /** adds a vertex on the current vertical edge */
    int add_z_vertex();
    /** adds a vertex inside the current cube */
    int add_c_vertex();

    /**
    * interpolates the horizontal gradient of the implicit function at the lower vertex of the specified cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    real get_x_grad( const int i, const int j, const int k ) const; 
    /**
    * interpolates the longitudinal gradient of the implicit function at the lower vertex of the specified cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    real get_y_grad( const int i, const int j, const int k ) const; 
    /**
    * interpolates the vertical gradient of the implicit function at the lower vertex of the specified cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    real get_z_grad( const int i, const int j, const int k ) const; 

    /**
    * accesses the pre-computed vertex index on the lower horizontal edge of a specific cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline int   get_x_vert( const int i, const int j, const int k ) const { return _x_verts[ i + j*_size_x + k*_size_x*_size_y]; }
    /**
    * accesses the pre-computed vertex index on the lower longitudinal edge of a specific cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline int   get_y_vert( const int i, const int j, const int k ) const { return _y_verts[ i + j*_size_x + k*_size_x*_size_y]; }
    /**
    * accesses the pre-computed vertex index on the lower vertical edge of a specific cube
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline int   get_z_vert( const int i, const int j, const int k ) const { return _z_verts[ i + j*_size_x + k*_size_x*_size_y]; }

    /**
    * sets the pre-computed vertex index on the lower horizontal edge of a specific cube
    * \param val the index of the new vertex
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline void  set_x_vert( const int val, const int i, const int j, const int k ) { _x_verts[ i + j*_size_x + k*_size_x*_size_y] = val; }
    /**
    * sets the pre-computed vertex index on the lower longitudinal edge of a specific cube
    * \param val the index of the new vertex
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline void  set_y_vert( const int val, const int i, const int j, const int k ) { _y_verts[ i + j*_size_x + k*_size_x*_size_y] = val; }
    /**
    * sets the pre-computed vertex index on the lower vertical edge of a specific cube
    * \param val the index of the new vertex
    * \param i abscisse of the cube
    * \param j ordinate of the cube
    * \param k height of the cube
    */
    inline void  set_z_vert( const int val, const int i, const int j, const int k ) { _z_verts[ i + j*_size_x + k*_size_x*_size_y] = val; }

    /** prints cube for debug */
    void    print_cube();

    /** checks the size of the connectivity vectors and allocates more space if required */
    void resizeAndAllocateConnectivity(int index);

    //-----------------------------------------------------------------------------
    // Elements
  protected :
    bool      _originalMC;   /**< selects wether the algorithm will use the enhanced topologically controlled lookup table or the original MarchingCubes */
    bool      _ext_data  ;   /**< selects wether to allocate data or use data from another class */

    bool      _computeNormals; /**< selects wether to compute normals or not */

    int       _size_x    ;  /**< width  of the grid */
    int       _size_y    ;  /**< depth  of the grid */
    int       _size_z    ;  /**< height of the grid */
    real     *_data      ;  /**< implicit function values sampled on the grid */ 

    int      *_x_verts   ;  /**< pre-computed vertex indices on the lower horizontal   edge of each cube */
    int      *_y_verts   ;  /**< pre-computed vertex indices on the lower longitudinal edge of each cube */
    int      *_z_verts   ;  /**< pre-computed vertex indices on the lower vertical     edge of each cube */

    int       _nverts    ;  /**< number of allocated vertices  in the vertex   buffer */
    int       _ntrigs    ;  /**< number of allocated triangles in the triangle buffer */
    int       _Nverts    ;  /**< allocated size of the vertex   buffer  - buffer might have fewer elements*/
    int       _Ntrigs    ;  /**< allocated size of the triangle buffer  - buffer might have fewer elements*/


    mitk::MeshData * m_MeshDataExt;  /**< externally allocated data structure that contains the mesh that we're working on*/

    int       _i         ;  /**< abscisse of the active cube */
    int       _j         ;  /**< height of the active cube */
    int       _k         ;  /**< ordinate of the active cube */

    real      _cube[8]   ;  /**< values of the implicit function on the active cube */ 
    uchar     _lut_entry ;  /**< cube sign representation in [0..255] */
    uchar     _case      ;  /**< case of the active cube in [0..15] */
    uchar     _config    ;  /**< configuration of the active cube */
    uchar     _subconfig ;  /**< subconfiguration of the active cube */
  };
  //_____________________________________________________________________________
} // namespace mitk

#endif // mitkNifTKCMC33_h