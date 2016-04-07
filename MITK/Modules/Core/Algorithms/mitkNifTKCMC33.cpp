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

#include <math.h>
#include <time.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <iostream> 
#include <fstream>
#include <algorithm>

#include "mitkNifTKCMC33.h"
#include "mitkNifTKCMC33LookUpTable.h"

// step size of the arrays of vertices and triangles
#define ALLOC_SIZE 65536

namespace mitk {
//_____________________________________________________________________________
// print cube for debug
void CMC33::print_cube() 
{ 
  printf("\t%f %f %f %f %f %f %f %f\n", _cube[0], _cube[1], _cube[2], _cube[3], _cube[4], _cube[5], _cube[6], _cube[7]);
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Constructor
CMC33::CMC33(const int size_x /*= -1*/, const int size_y /*= -1*/, const int size_z /*= -1*/) :
//-----------------------------------------------------------------------------
_originalMC(false),
  _ext_data (false),
  _computeNormals(false),
  _size_x (size_x),
  _size_y (size_y),
  _size_z (size_z),
  _data ((real *)NULL),
  _x_verts ((int *)NULL),
  _y_verts ((int *)NULL),
  _z_verts ((int *)NULL),
  _nverts (0),
  _ntrigs (0),
  _Nverts (0),
  _Ntrigs (0),
  m_MeshDataExt(0)
{
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Destructor
CMC33::~CMC33()
//-----------------------------------------------------------------------------
{
  clean_all();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// main algorithm
void CMC33::run(real iso)
//-----------------------------------------------------------------------------
{
  clock_t time = clock();

  compute_intersection_points(iso);

  for(_k = 0; _k < _size_z-1; _k++)
    for(_j = 0; _j < _size_y-1; _j++)
      for(_i = 0; _i < _size_x-1; _i++)
      {
        _lut_entry = 0;
        for(int p = 0; p < 8; ++p)
        {
          _cube[p] = get_data(_i+((p^(p>>1))&1), _j+((p>>1)&1), _k+((p>>2)&1)) - iso;

          if(fabs(_cube[p]) < FLT_EPSILON)
            _cube[p] = FLT_EPSILON;

          if(_cube[p] > 0)
            _lut_entry += 1 << p;
        }

        process_cube();
      }


  m_MeshDataExt->m_Vertices.resize(_nverts);
  m_MeshDataExt->m_Triangles.resize(_ntrigs);
  m_MeshDataExt->m_VertexToTriangleIndices.resize(_nverts);
  m_MeshDataExt->m_VertexToVertexIndices.resize(_nverts);

  printf("New Marching Cubes 33:ran in %lf secs.\n", (double) (clock() - time) / CLOCKS_PER_SEC);
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// init temporary structures (must set sizes before call)
void CMC33::init_temps()
//-----------------------------------------------------------------------------
{
  if(!_ext_data)
    _data = new real [_size_x * _size_y * _size_z];
  
  _x_verts = new int [_size_x * _size_y * _size_z];
  _y_verts = new int [_size_x * _size_y * _size_z];
  _z_verts = new int [_size_x * _size_y * _size_z];

  memset(_x_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
  memset(_y_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
  memset(_z_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// init all structures (must set sizes before call)
void CMC33::init_all ()
//-----------------------------------------------------------------------------
{
  if (m_MeshDataExt == 0)
  {
    std::cout <<"C-MC33 Extractor cannot be initialized: output data was not set!\n";
    return;
  }

  init_temps();

  _nverts = _ntrigs = 0;
  _Nverts = _Ntrigs = ALLOC_SIZE;

  m_MeshDataExt->m_Vertices.resize(_Nverts);
  m_MeshDataExt->m_Triangles.resize(_Ntrigs);
  m_MeshDataExt->m_VertexToVertexIndices.resize(_Nverts);
  m_MeshDataExt->m_VertexToTriangleIndices.resize(_Nverts);
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// clean temporary structures
void CMC33::clean_temps()
  //-----------------------------------------------------------------------------
{
  if(!_ext_data)
    delete [] _data;

  delete [] _x_verts;
  delete [] _y_verts;
  delete [] _z_verts;

  if(!_ext_data)
    _data = (real*)NULL;

  _x_verts = (int*)NULL;
  _y_verts = (int*)NULL;
  _z_verts = (int*)NULL;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// clean all structures
void CMC33::clean_all()
  //-----------------------------------------------------------------------------
{
  clean_temps();

  _nverts = _ntrigs = 0;
  _Nverts = _Ntrigs = 0;

  _size_x = _size_y = _size_z = -1;
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// clean mesh and restart new one
void CMC33::restart()
  //-----------------------------------------------------------------------------
{
  m_MeshDataExt->m_Vertices.clear();
  m_MeshDataExt->m_Triangles.clear();
  m_MeshDataExt->m_VertexToTriangleIndices.clear();
  m_MeshDataExt->m_VertexToVertexIndices.clear();

  _nverts = _ntrigs = 0;
  _Nverts = _Ntrigs = ALLOC_SIZE;
  memset(_x_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
  memset(_y_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
  memset(_z_verts, -1, _size_x * _size_y * _size_z * sizeof(int));
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// Compute the intersection points
void CMC33::compute_intersection_points(real iso)
//-----------------------------------------------------------------------------
{
  for(_k = 0; _k < _size_z; _k++)
    for(_j = 0; _j < _size_y; _j++)
      for(_i = 0; _i < _size_x; _i++)
      {
        _cube[0] = get_data(_i, _j, _k) - iso;
        if(_i < _size_x - 1) 
          _cube[1] = get_data(_i+1, _j , _k) - iso;
        else 
          _cube[1] = _cube[0];

        if(_j < _size_y - 1)
          _cube[3] = get_data(_i ,_j+1, _k) - iso;
        else
          _cube[3] = _cube[0];

        if(_k < _size_z - 1)
          _cube[4] = get_data(_i , _j ,_k+1) - iso;
        else
          _cube[4] = _cube[0];

        if(fabs(_cube[0]) < FLT_EPSILON)
          _cube[0] = FLT_EPSILON;

        if(fabs(_cube[1]) < FLT_EPSILON)
          _cube[1] = FLT_EPSILON;

        if(fabs(_cube[3]) < FLT_EPSILON)
          _cube[3] = FLT_EPSILON;

        if(fabs(_cube[4]) < FLT_EPSILON)
          _cube[4] = FLT_EPSILON;

        if(_cube[0] < 0)
        {
          if(_cube[1] > 0)
            set_x_vert(add_x_vertex(), _i,_j,_k);

          if(_cube[3] > 0)
            set_y_vert(add_y_vertex(), _i,_j,_k);

          if(_cube[4] > 0)
            set_z_vert(add_z_vertex(), _i,_j,_k);
        }
        else
        {
          if(_cube[1] < 0)
            set_x_vert(add_x_vertex(), _i,_j,_k);

          if(_cube[3] < 0)
            set_y_vert(add_y_vertex(), _i,_j,_k);

          if(_cube[4] < 0)
            set_z_vert(add_z_vertex(), _i,_j,_k);
        }
      }
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// Test a face
// if face>0 return true if the face contains a part of the surface
bool CMC33::test_face(schar face)
  //-----------------------------------------------------------------------------
{
  real A,B,C,D;

  switch(face)
  {
  case -1: 
  case 1 : 
    A = _cube[0];
    B = _cube[4];
    C = _cube[5];
    D = _cube[1];
    break;
  case -2:
  case 2 :
    A = _cube[1];
    B = _cube[5];
    C = _cube[6];
    D = _cube[2];
    break;
  case -3:
  case 3 :
    A = _cube[2];
    B = _cube[6];
    C = _cube[7];
    D = _cube[3];
    break;
  case -4:
  case 4 :
    A = _cube[3];
    B = _cube[7];
    C = _cube[4];
    D = _cube[0];
    break;
  case -5:
  case 5 :
    A = _cube[0];
    B = _cube[3];
    C = _cube[2];
    D = _cube[1];
    break;
  case -6:
  case 6 :
    A = _cube[4];
    B = _cube[7];
    C = _cube[6];
    D = _cube[5];
    break;
  default: 
    printf("Invalid face code %d\n", face);
    print_cube(); 
    A = B = C = D = 0;
    break;
  };

  if(fabs(A*C - B*D) < FLT_EPSILON)
    return face >= 0;

  return face * A * (A*C - B*D) >= 0; // face and A invert signs
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Test the interior of a cube
// if s == 7, return true if the interior is empty
// if s ==-7, return false if the interior is empty
bool CMC33::test_interior(schar s)
  //-----------------------------------------------------------------------------
{
  real t, At=0, Bt=0, Ct=0, Dt=0, a, b;
  char test = 0;
  char edge = -1; // reference edge of the triangulation

  switch(_case)
  {
  case 4 :
  case 10 :
    a = (_cube[4] - _cube[0]) * (_cube[6] - _cube[2]) - (_cube[7] - _cube[3]) * (_cube[5] - _cube[1]);
    b = _cube[2] * (_cube[4] - _cube[0]) + _cube[0] * (_cube[6] - _cube[2])
      - _cube[1] * (_cube[7] - _cube[3]) - _cube[3] * (_cube[5] - _cube[1]);
    t = - b / (2*a);
    if(t<0 || t>1) return s>0;

    At = _cube[0] + (_cube[4] - _cube[0]) * t;
    Bt = _cube[3] + (_cube[7] - _cube[3]) * t;
    Ct = _cube[2] + (_cube[6] - _cube[2]) * t;
    Dt = _cube[1] + (_cube[5] - _cube[1]) * t;
    break;

  case 6 :
  case 7 :
  case 12 :
  case 13 :
    switch(_case)
    {
    case 6 : edge = test6 [_config][2]; break;
    case 7 : edge = test7 [_config][4]; break;
    case 12 : edge = test12[_config][3]; break;
    case 13 : edge = tiling13_5_1[_config][_subconfig][0]; break;
    }
    switch(edge)
    {
    case 0 :
      t = _cube[0] / (_cube[0] - _cube[1]);
      At = 0;
      Bt = _cube[3] + (_cube[2] - _cube[3]) * t;
      Ct = _cube[7] + (_cube[6] - _cube[7]) * t;
      Dt = _cube[4] + (_cube[5] - _cube[4]) * t;
      break;
    case 1 :
      t = _cube[1] / (_cube[1] - _cube[2]);
      At = 0;
      Bt = _cube[0] + (_cube[3] - _cube[0]) * t;
      Ct = _cube[4] + (_cube[7] - _cube[4]) * t;
      Dt = _cube[5] + (_cube[6] - _cube[5]) * t;
      break;
    case 2 :
      t = _cube[2] / (_cube[2] - _cube[3]);
      At = 0;
      Bt = _cube[1] + (_cube[0] - _cube[1]) * t;
      Ct = _cube[5] + (_cube[4] - _cube[5]) * t;
      Dt = _cube[6] + (_cube[7] - _cube[6]) * t;
      break;
    case 3 :
      t = _cube[3] / (_cube[3] - _cube[0]);
      At = 0;
      Bt = _cube[2] + (_cube[1] - _cube[2]) * t;
      Ct = _cube[6] + (_cube[5] - _cube[6]) * t;
      Dt = _cube[7] + (_cube[4] - _cube[7]) * t;
      break;
    case 4 :
      t = _cube[4] / (_cube[4] - _cube[5]);
      At = 0;
      Bt = _cube[7] + (_cube[6] - _cube[7]) * t;
      Ct = _cube[3] + (_cube[2] - _cube[3]) * t;
      Dt = _cube[0] + (_cube[1] - _cube[0]) * t;
      break;
    case 5 :
      t = _cube[5] / (_cube[5] - _cube[6]);
      At = 0;
      Bt = _cube[4] + (_cube[7] - _cube[4]) * t;
      Ct = _cube[0] + (_cube[3] - _cube[0]) * t;
      Dt = _cube[1] + (_cube[2] - _cube[1]) * t;
      break;
    case 6 :
      t = _cube[6] / (_cube[6] - _cube[7]);
      At = 0;
      Bt = _cube[5] + (_cube[4] - _cube[5]) * t;
      Ct = _cube[1] + (_cube[0] - _cube[1]) * t;
      Dt = _cube[2] + (_cube[3] - _cube[2]) * t;
      break;
    case 7 :
      t = _cube[7] / (_cube[7] - _cube[4]);
      At = 0;
      Bt = _cube[6] + (_cube[5] - _cube[6]) * t;
      Ct = _cube[2] + (_cube[1] - _cube[2]) * t;
      Dt = _cube[3] + (_cube[0] - _cube[3]) * t;
      break;
    case 8 :
      t = _cube[0] / (_cube[0] - _cube[4]);
      At = 0;
      Bt = _cube[3] + (_cube[7] - _cube[3]) * t;
      Ct = _cube[2] + (_cube[6] - _cube[2]) * t;
      Dt = _cube[1] + (_cube[5] - _cube[1]) * t;
      break;
    case 9 :
      t = _cube[1] / (_cube[1] - _cube[5]);
      At = 0;
      Bt = _cube[0] + (_cube[4] - _cube[0]) * t;
      Ct = _cube[3] + (_cube[7] - _cube[3]) * t;
      Dt = _cube[2] + (_cube[6] - _cube[2]) * t;
      break;
    case 10 :
      t = _cube[2] / (_cube[2] - _cube[6]);
      At = 0;
      Bt = _cube[1] + (_cube[5] - _cube[1]) * t;
      Ct = _cube[0] + (_cube[4] - _cube[0]) * t;
      Dt = _cube[3] + (_cube[7] - _cube[3]) * t;
      break;
    case 11 :
      t = _cube[3] / (_cube[3] - _cube[7]);
      At = 0;
      Bt = _cube[2] + (_cube[6] - _cube[2]) * t;
      Ct = _cube[1] + (_cube[5] - _cube[1]) * t;
      Dt = _cube[0] + (_cube[4] - _cube[0]) * t;
      break;
    default : printf("Invalid edge %d\n", edge); print_cube(); break;
    }
    break;

  default : printf("Invalid ambiguous case %d\n", _case); print_cube(); break;
  }

  if(At >= 0)
    test ++;
  if(Bt >= 0)
    test += 2;
  if(Ct >= 0)
    test += 4;
  if(Dt >= 0)
    test += 8;
  
  switch(test)
  {
    case 0 : return s>0;
    case 1 : return s>0;
    case 2 : return s>0;
    case 3 : return s>0;
    case 4 : return s>0;
    case 5 : if(At * Ct - Bt * Dt < FLT_EPSILON) return s>0; break;
    case 6 : return s>0;
    case 7 : return s<0;
    case 8 : return s>0;
    case 9 : return s>0;
    case 10 : if(At * Ct - Bt * Dt >= FLT_EPSILON) return s>0; break;
    case 11 : return s<0;
    case 12 : return s>0;
    case 13 : return s<0;
    case 14 : return s<0;
    case 15 : return s<0;
  }

  return s<0;
}
//_____________________________________________________________________________


//____________________________________________________________________________

bool CMC33::modified_test_interior(schar s)
  //-----------------------------------------------------------------------------
{
  char edge = -1;
  int amb_face;

  int inter_amb = 0;

  switch (_case)
  {
  case 4:
    amb_face = 1;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 2;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 5;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0) 
      return false;
    else 
      return true;
    break;
  case 6:
    amb_face = abs(test6[_config][0]);
    edge = interior_ambiguity(amb_face, s);
    inter_amb = interior_ambiguity_verification(edge);

    if (inter_amb == 0)
      return false;
    else
      return true;
    break;
  case 7:
    s = s * -1;

    amb_face = 1;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 2;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 5;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0) 
      return false;
    else 
      return true;
    break;
  case 10:
    amb_face = abs(test10[_config][0]);

    edge = interior_ambiguity(amb_face, s);
    inter_amb = interior_ambiguity_verification(edge);

    if (inter_amb == 0)
      return false;
    else
      return true;
    break;
  case 12:
    amb_face = abs(test12[_config][0]);
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = abs(test12[_config][1]);
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0)
      return false;
    else
      return true;
    break;
  }
  
  //std::cout <<"Should never get here!\n";
  return false;
}
//________________________________________________________________________________________________________


//________________________________________________________________________________________________________
int CMC33::interior_ambiguity(int amb_face, int s)
{
  int edge;

  switch (amb_face)
  {
  case 1:
  case 3:
    if (((_cube[1] * s) > 0) && ((_cube[7] * s) > 0))
      edge = 4;
    if (((_cube[0] * s) > 0) && ((_cube[6] * s) > 0))
      edge = 5;
    if (((_cube[3] * s) > 0) && ((_cube[5] * s) > 0))
      edge = 6;
    if (((_cube[2] * s) > 0) && ((_cube[4] * s) > 0))
      edge = 7;
    break;

  case 2:
  case 4:
    if (((_cube[1] * s) > 0) && ((_cube[7] * s) > 0))
      edge = 0;
    if (((_cube[2] * s) > 0) && ((_cube[4] * s) > 0))
      edge = 1;
    if (((_cube[3] * s) > 0) && ((_cube[5] * s) > 0))
      edge = 2;
    if (((_cube[0] * s) > 0) && ((_cube[6] * s) > 0))
      edge = 3;
    break;

  case 5:
  case 6:
  case 0:
    if (((_cube[0] * s) > 0) && ((_cube[6] * s) > 0))
      edge = 8;
    if (((_cube[1] * s) > 0) && ((_cube[7] * s) > 0))
      edge = 9;
    if (((_cube[2] * s) > 0) && ((_cube[4] * s) > 0))
      edge = 10;
    if (((_cube[3] * s) > 0) && ((_cube[5] * s) > 0))
      edge = 11;

    break;
  }

  return edge;
}
//-----------------------------------------------------------------------------
int CMC33::interior_ambiguity_verification(int edge)
{
  real t, At = 0, Bt = 0, Ct = 0, Dt = 0, a = 0, b = 0;
  real verify;

  switch (edge) 
  {

  case 0:
    a = (_cube[0] - _cube[1]) * (_cube[7] - _cube[6])
      - (_cube[4] - _cube[5]) * (_cube[3] - _cube[2]);
    
    b = _cube[6] * (_cube[0] - _cube[1]) 
      + _cube[1] * (_cube[7] - _cube[6])
      - _cube[2] * (_cube[4] - _cube[5])
      - _cube[5] * (_cube[3] - _cube[2]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[1] + (_cube[0] - _cube[1]) * t;
    Bt = _cube[5] + (_cube[4] - _cube[5]) * t;
    Ct = _cube[6] + (_cube[7] - _cube[6]) * t;
    Dt = _cube[2] + (_cube[3] - _cube[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;
  case 1:
    a = (_cube[3] - _cube[2]) * (_cube[4] - _cube[5])
      - (_cube[0] - _cube[1]) * (_cube[7] - _cube[6]);

    b = _cube[5] * (_cube[3] - _cube[2])
      + _cube[2] * (_cube[4] - _cube[5])
      - _cube[6] * (_cube[0] - _cube[1])
      - _cube[1] * (_cube[7] - _cube[6]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[2] + (_cube[3] - _cube[2]) * t;
    Bt = _cube[1] + (_cube[0] - _cube[1]) * t;
    Ct = _cube[5] + (_cube[4] - _cube[5]) * t;
    Dt = _cube[6] + (_cube[7] - _cube[6]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 2:
    a = (_cube[2] - _cube[3]) * (_cube[5] - _cube[4])
      - (_cube[6] - _cube[7]) * (_cube[1] - _cube[0]);

    b = _cube[4] * (_cube[2] - _cube[3])
      + _cube[3] * (_cube[5] - _cube[4])
      - _cube[0] * (_cube[6] - _cube[7])
      - _cube[7] * (_cube[1] - _cube[0]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[3] + (_cube[2] - _cube[3]) * t;
    Bt = _cube[7] + (_cube[6] - _cube[7]) * t;
    Ct = _cube[4] + (_cube[5] - _cube[4]) * t;
    Dt = _cube[0] + (_cube[1] - _cube[0]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 3:
    a = (_cube[1] - _cube[0]) * (_cube[6] - _cube[7])
      - (_cube[2] - _cube[3]) * (_cube[5] - _cube[4]);

    b = _cube[7] * (_cube[1] - _cube[0])
      + _cube[0] * (_cube[6] - _cube[7])
      - _cube[4] * (_cube[2] - _cube[3])
      - _cube[3] * (_cube[5] - _cube[4]);
    
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[0] + (_cube[1] - _cube[0]) * t;
    Bt = _cube[3] + (_cube[2] - _cube[3]) * t;
    Ct = _cube[7] + (_cube[6] - _cube[7]) * t;
    Dt = _cube[4] + (_cube[5] - _cube[4]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 4:

    a = (_cube[2] - _cube[1]) * (_cube[7] - _cube[4])
      - (_cube[3] - _cube[0]) * (_cube[6] - _cube[5]);

    b = _cube[4] * (_cube[2] - _cube[1])
      + _cube[1] * (_cube[7] - _cube[4])
      - _cube[5] * (_cube[3] - _cube[0])
      - _cube[0] * (_cube[6] - _cube[5]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[1] + (_cube[2] - _cube[1]) * t;
    Bt = _cube[0] + (_cube[3] - _cube[0]) * t;
    Ct = _cube[4] + (_cube[7] - _cube[4]) * t;
    Dt = _cube[5] + (_cube[6] - _cube[5]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 5:

    a = (_cube[3] - _cube[0]) * (_cube[6] - _cube[5])
      - (_cube[2] - _cube[1]) * (_cube[7] - _cube[4]);

    b = _cube[5] * (_cube[3] - _cube[0])
      + _cube[0] * (_cube[6] - _cube[5])
      - _cube[4] * (_cube[2] - _cube[1])
      - _cube[1] * (_cube[7] - _cube[4]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[0] + (_cube[3] - _cube[0]) * t;
    Bt = _cube[1] + (_cube[2] - _cube[1]) * t;
    Ct = _cube[5] + (_cube[6] - _cube[5]) * t;
    Dt = _cube[4] + (_cube[7] - _cube[4]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 6:
    a = (_cube[0] - _cube[3]) * (_cube[5] - _cube[6])
      - (_cube[4] - _cube[7]) * (_cube[1] - _cube[2]);

    b = _cube[6] * (_cube[0] - _cube[3])
      + _cube[3] * (_cube[5] - _cube[6])
      - _cube[2] * (_cube[4] - _cube[7])
      - _cube[7] * (_cube[1] - _cube[2]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[3] + (_cube[0] - _cube[3]) * t;
    Bt = _cube[7] + (_cube[4] - _cube[7]) * t;
    Ct = _cube[6] + (_cube[5] - _cube[6]) * t;
    Dt = _cube[2] + (_cube[1] - _cube[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 7:
    a = (_cube[1] - _cube[2]) * (_cube[4] - _cube[7])
      - (_cube[0] - _cube[3]) * (_cube[5] - _cube[6]);

    b = _cube[7] * (_cube[1] - _cube[2])
      + _cube[2] * (_cube[4] - _cube[7])
      - _cube[6] * (_cube[0] - _cube[3])
      - _cube[3] * (_cube[5] - _cube[6]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[2] + (_cube[1] - _cube[2]) * t;
    Bt = _cube[3] + (_cube[0] - _cube[3]) * t;
    Ct = _cube[7] + (_cube[4] - _cube[7]) * t;
    Dt = _cube[6] + (_cube[5] - _cube[6]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 8:
    a = (_cube[4] - _cube[0]) * (_cube[6] - _cube[2])
      - (_cube[7] - _cube[3]) * (_cube[5] - _cube[1]);

    b = _cube[2] * (_cube[4] - _cube[0])
      + _cube[0] * (_cube[6] - _cube[2])
      - _cube[1] * (_cube[7] - _cube[3])
      - _cube[3] * (_cube[5] - _cube[1]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[0] + (_cube[4] - _cube[0]) * t;
    Bt = _cube[3] + (_cube[7] - _cube[3]) * t;
    Ct = _cube[2] + (_cube[6] - _cube[2]) * t;
    Dt = _cube[1] + (_cube[5] - _cube[1]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 9:
    a = (_cube[5] - _cube[1]) * (_cube[7] - _cube[3])
      - (_cube[4] - _cube[0]) * (_cube[6] - _cube[2]);

    b = _cube[3] * (_cube[5] - _cube[1])
      + _cube[1] * (_cube[7] - _cube[3])
      - _cube[2] * (_cube[4] - _cube[0])
      - _cube[0] * (_cube[6] - _cube[2]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[1] + (_cube[5] - _cube[1]) * t;
    Bt = _cube[0] + (_cube[4] - _cube[0]) * t;
    Ct = _cube[3] + (_cube[7] - _cube[3]) * t;
    Dt = _cube[2] + (_cube[6] - _cube[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 10:
    a = (_cube[6] - _cube[2]) * (_cube[4] - _cube[0])
      - (_cube[5] - _cube[1]) * (_cube[7] - _cube[3]);

    b = _cube[0] * (_cube[6] - _cube[2])
      + _cube[2] * (_cube[4] - _cube[0])
      - _cube[3] * (_cube[5] - _cube[1])
      - _cube[1] * (_cube[7] - _cube[3]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[2] + (_cube[6] - _cube[2]) * t;
    Bt = _cube[1] + (_cube[5] - _cube[1]) * t;
    Ct = _cube[0] + (_cube[4] - _cube[0]) * t;
    Dt = _cube[3] + (_cube[7] - _cube[3]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 11:
    a = (_cube[7] - _cube[3]) * (_cube[5] - _cube[1])
      - (_cube[6] - _cube[2]) * (_cube[4] - _cube[0]);

    b = _cube[1] * (_cube[7] - _cube[3])
      + _cube[3] * (_cube[5] - _cube[1])
      - _cube[0] * (_cube[6] - _cube[2])
      - _cube[2] * (_cube[4] - _cube[0]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = _cube[3] + (_cube[7] - _cube[3]) * t;
    Bt = _cube[2] + (_cube[6] - _cube[2]) * t;
    Ct = _cube[1] + (_cube[5] - _cube[1]) * t;
    Dt = _cube[0] + (_cube[4] - _cube[0]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;
  }

  //std::cout <<"Should never get here!\n";
  return -1;
}

//_____________________________________________________________________________
// NEWS INTERIOR TEST FOR CASE 13
// Return true if the interior is empty(two faces)

bool CMC33::interior_test_case13()
{
  real t1, t2, At1 = 0, Bt1 = 0, Ct1 = 0, Dt1 = 0, At2 = 0, Bt2 = 0, Ct2 = 0, Dt2 = 0, a = 0, b = 0, c = 0;

  a = (_cube[0] - _cube[1]) * (_cube[7] - _cube[6])
    - (_cube[4] - _cube[5]) * (_cube[3] - _cube[2]);

  b = _cube[6] * (_cube[0] - _cube[1])
    + _cube[1] * (_cube[7] - _cube[6])
    - _cube[2] * (_cube[4] - _cube[5])
    - _cube[5] * (_cube[3] - _cube[2]);

  c = _cube[1]*_cube[6] - _cube[5]*_cube[2];

  double delta = b*b - 4*a*c;

  t1 = (-b + sqrt(delta))/(2*a);
  t2 = (-b - sqrt(delta))/(2*a);

  if ((t1 < 1)&&(t1>0) &&(t2 < 1)&&(t2 > 0))
  {
    At1 = _cube[1] + (_cube[0] - _cube[1]) * t1;
    Bt1 = _cube[5] + (_cube[4] - _cube[5]) * t1;
    Ct1 = _cube[6] + (_cube[7] - _cube[6]) * t1;
    Dt1 = _cube[2] + (_cube[3] - _cube[2]) * t1;

    float x1 = (At1 - Dt1)/(At1 + Ct1 - Bt1 - Dt1);
    float y1 = (At1 - Bt1)/(At1 + Ct1 - Bt1 - Dt1);

    At2 = _cube[1] + (_cube[0] - _cube[1]) * t2;
    Bt2 = _cube[5] + (_cube[4] - _cube[5]) * t2;
    Ct2 = _cube[6] + (_cube[7] - _cube[6]) * t2;
    Dt2 = _cube[2] + (_cube[3] - _cube[2]) * t2;

    float x2 = (At2 - Dt2)/(At2 + Ct2 - Bt2 - Dt2);
    float y2 = (At2 - Bt2)/(At2 + Ct2 - Bt2 - Dt2);

    if ((x1 < 1)&&(x1>0) &&(x2 < 1)&&(x2 > 0)&&(y1 < 1)&&(y1>0) &&(y2 < 1)&&(y2 > 0))
      return false;
    else
      return true;
  }

  else return true;
}

//--------------------------------------------------------------------------------------------------------------------
// control the tunnel orientation triangulation
int tunnelOrientation = 0;

bool CMC33::interior_test_case13_2(float isovalue)
{

  double critival_point_value1, critival_point_value2;

  double a = - _cube[0] + _cube[1] + _cube[3] - _cube[2] + _cube[4] - _cube[5] - _cube[7] + _cube[6];
  double b = _cube[0] - _cube[1] - _cube[3] + _cube[2];
  double c = _cube[0] - _cube[1] - _cube[4] + _cube[5];
  double d = _cube[0] - _cube[3] - _cube[4] + _cube[7];
  double e = -_cube[0] + _cube[1];
  double f = -_cube[0] + _cube[3];
  double g = -_cube[0] + _cube[4];
  double h = _cube[0];

  double x1, y1, z1, x2, y2, z2;
  int numbercritivalpoints = 0;

  double dx = b * c - a * e;
  double dy = b * d - a * f;
  double dz = c * d - a * g;

  if (dx != 0.0f && dy != 0.0f && dz != 0.0f) 
  {
    if (dx * dy * dz < 0)
      return true;

    double disc = sqrt(dx * dy * dz);

    x1 = (-d * dx - disc) / (a * dx);
    y1 = (-c * dy - disc) / (a * dy);
    z1 = (-b * dz - disc) / (a * dz);

    if ((x1 > 0) && (x1 < 1) && (y1 > 0) && (y1 < 1) && (z1 > 0) && (z1 < 1)) 
    {
        numbercritivalpoints++;

        critival_point_value1 = a * x1 * y1 * z1 + b * x1 * y1 + c * x1 * z1
          + d * y1 * z1 + e * x1 + f * y1 + g * z1 + h - isovalue;
    }

    x2 = (-d * dx + disc) / (a * dx);
    y2 = (-c * dy + disc) / (a * dy);
    z2 = (-b * dz + disc) / (a * dz);

    if ((x2 > 0) && (x2 < 1) && (y2 > 0) && (y2 < 1) && (z2 > 0) && (z2 < 1))
    {
      numbercritivalpoints++;

        critival_point_value2 = a * x2 * y2 * z2 + b * x2 * y2 + c * x2 * z2
          + d * y2 * z2 + e * x2 + f * y2 + g * z2 + h - isovalue;
    }

    if (numbercritivalpoints < 2)
      return true;
    else
    {
      if ((critival_point_value1 * critival_point_value2 > 0))
      {
        if (critival_point_value1 > 0)
          tunnelOrientation = 1;
        else
          tunnelOrientation = -1;
      }

      return critival_point_value1 * critival_point_value2 < 0;
    }

  } 
  else
    return true;
}

//_____________________________________________________________________________
// Process a unit cube
void CMC33::process_cube()
  //-----------------------------------------------------------------------------
{
  if(_originalMC)
  {
    char nt = 0;
    while(casesClassic[_lut_entry][3*nt] != -1)
      nt++;
    add_triangle(casesClassic[_lut_entry], nt);
    return;
  }

  int v12 = -1;
  _case = cases[_lut_entry][0];
  _config = cases[_lut_entry][1];
  _subconfig = 0;

  switch(_case)
  {
  case 0 :
    break;

  case 1 :
    add_triangle(tiling1[_config], 1);
    break;

  case 2 :
    add_triangle(tiling2[_config], 2);
    break;

  case 3 :
    if(test_face(test3[_config]))
      add_triangle(tiling3_2[_config], 4); // 3.2
    else
      add_triangle(tiling3_1[_config], 2); // 3.1
    break;

  case 4 :
    if (modified_test_interior(test4[_config]))
      add_triangle(tiling4_1[_config], 2); // 4.1.1
    else
      add_triangle(tiling4_2[_config], 6); // 4.1.2
    break;

  case 5 :
    add_triangle(tiling5[_config], 3);
    break;

  case 6 :
    if(test_face(test6[_config][0]))
      add_triangle(tiling6_2[_config], 5); // 6.2
    else
    {
      if (modified_test_interior(test6[_config][1]))
        add_triangle(tiling6_1_1[_config], 3); // 6.1.1
      else
      {
        v12 = add_c_vertex(); //this line is not in the 2008 version
        add_triangle(tiling6_1_2[_config], 9 , v12); // 6.1.2
      }
    }
    break;

  case 7 :
    if(test_face(test7[_config][0]))
      _subconfig += 1;
    if(test_face(test7[_config][1]))
      _subconfig += 2;
    if(test_face(test7[_config][2]))
      _subconfig += 4;
    switch(_subconfig)
    {
    case 0 :
      add_triangle(tiling7_1[_config], 3);
      break;
    case 1 :
      add_triangle(tiling7_2[_config][0], 5);
      break;
    case 2 :
      add_triangle(tiling7_2[_config][1], 5);
      break;
    case 3 :
      v12 = add_c_vertex();
      add_triangle(tiling7_3[_config][0], 9, v12);
      break;
    case 4 :
      add_triangle(tiling7_2[_config][2], 5);
      break;
    case 5 :
      v12 = add_c_vertex();
      add_triangle(tiling7_3[_config][1], 9, v12);
      break;
    case 6 :
      v12 = add_c_vertex();
      add_triangle(tiling7_3[_config][2], 9, v12);
      break;
    case 7:
      if (modified_test_interior(test7[_config][3]))
        add_triangle(tiling7_4_1[_config], 5);
      else
        add_triangle(tiling7_4_2[_config], 9);
      break;
    }
    break;
  case 8 :
    add_triangle(tiling8[_config], 2);
    break;

  case 9 :
    add_triangle(tiling9[_config], 4);
    break;

  case 10 :
    if(test_face(test10[_config][0]))
    {
      /*
      if(test_face(test10[_config][1]))
      add_triangle(tiling10_1_1_[_config], 4); // 10.1.1
      */
      if (test_face(test10[_config][1]))
      {
        if (modified_test_interior(-test10[_config][2]))
          add_triangle(tiling10_1_1_[_config], 4); // 10.1.1
        else
          add_triangle(tiling10_1_2[5 - _config], 8); // 10.1.2

      }
      else
      {
        v12 = add_c_vertex();
        add_triangle(tiling10_2[_config], 8, v12); // 10.2
      }
    }
    else
    {
      if(test_face(test10[_config][1]))
      {
        v12 = add_c_vertex();
        add_triangle(tiling10_2_[_config], 8, v12); // 10.2
      }
      else
      {
        if (modified_test_interior(test10[_config][2]))
          add_triangle(tiling10_1_1[_config], 4); // 10.1.1
        else
          add_triangle(tiling10_1_2[_config], 8); // 10.1.2
      }
    }
    break;

  case 11 :
    add_triangle(tiling11[_config], 4);
    break;

  case 12 :
    if(test_face(test12[_config][0]))
    {
      if(test_face(test12[_config][1]))
        //add_triangle(tiling12_1_1_[_config], 4); // 12.1.1
      {
        if (modified_test_interior(-test12[_config][2]))
          add_triangle(tiling12_1_1_[_config], 4); // 12.1.1
        else
          add_triangle(tiling12_1_2[23 - _config], 8); // 12.1.2
      }
      else
      {
        v12 = add_c_vertex();
        add_triangle(tiling12_2[_config], 8, v12); // 12.2
      }
    }
    else
    {
      if(test_face(test12[_config][1]))
      {
        v12 = add_c_vertex();
        add_triangle(tiling12_2_[_config], 8, v12); // 12.2
      }
      else
      {
        if (modified_test_interior(test12[_config][2]))
          add_triangle(tiling12_1_1[_config], 4); // 12.1.1
        else
          add_triangle(tiling12_1_2[_config], 8); // 12.1.2
      }
    }
    break;

  case 13 :
    if(test_face(test13[_config][0]))
      _subconfig += 1;
    if(test_face(test13[_config][1]))
      _subconfig += 2;
    if(test_face(test13[_config][2]))
      _subconfig += 4;
    if(test_face(test13[_config][3]))
      _subconfig += 8;
    if(test_face(test13[_config][4]))
      _subconfig += 16;
    if(test_face(test13[_config][5]))
      _subconfig += 32;
    switch(subconfig13[_subconfig])
    {
    case 0 :/* 13.1 */
      add_triangle(tiling13_1[_config], 4);
      break;
    case 1 :/* 13.2 */
      add_triangle(tiling13_2[_config][0], 6);
      break;
    case 2 :/* 13.2 */
      add_triangle(tiling13_2[_config][1], 6);
      break;
    case 3 :/* 13.2 */
      add_triangle(tiling13_2[_config][2], 6);
      break;
    case 4 :/* 13.2 */
      add_triangle(tiling13_2[_config][3], 6);
      break;
    case 5 :/* 13.2 */
      add_triangle(tiling13_2[_config][4], 6);
      break;
    case 6 :/* 13.2 */
      add_triangle(tiling13_2[_config][5], 6);
      break;
    case 7 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][0], 10, v12);
      break;
    case 8 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][1], 10, v12);
      break;
    case 9 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][2], 10, v12);
      break;
    case 10 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][3], 10, v12);
      break;
    case 11 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][4], 10, v12);
      break;
    case 12 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][5], 10, v12);
      break;
    case 13 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][6], 10, v12);
      break;
    case 14 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][7], 10, v12);
      break;
    case 15 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][8], 10, v12);
      break;
    case 16 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][9], 10, v12);
      break;
    case 17 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][10], 10, v12);
      break;
    case 18 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3[_config][11], 10, v12);
      break;
    case 19 :/* 13.4 */
      v12 = add_c_vertex();
      add_triangle(tiling13_4[_config][0], 12, v12);
      break;
    case 20 :/* 13.4 */
      v12 = add_c_vertex();
      add_triangle(tiling13_4[_config][1], 12, v12);
      break;
    case 21 :/* 13.4 */
      v12 = add_c_vertex();
      add_triangle(tiling13_4[_config][2], 12, v12);
      break;
    case 22 :/* 13.4 */
      v12 = add_c_vertex();
      add_triangle(tiling13_4[_config][3], 12, v12);
      break;
    case 23 :/* 13.5 */
      /*
      _subconfig = 0;
      if(test_interior(test13[_config][6]))
      add_triangle(tiling13_5_1[_config][0], 6);
      else
      add_triangle(tiling13_5_2[_config][0], 10);
      break;
      */
      _subconfig = 0;
      if (_config == 0)
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[0][0], 6);
        else
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[0][0], 10);
          else
            add_triangle(tiling13_5_2[1][2], 10);
        }
      }
      else
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[1][0], 6);
        else
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[1][0], 10);
          else
            add_triangle(tiling13_5_2[0][2], 10);
        }
      }
      break;
    case 24 :/* 13.5 */
      /*
      _subconfig = 1;
      if(test_interior(test13[_config][6]))
      add_triangle(tiling13_5_1[_config][1], 6);
      else
      add_triangle(tiling13_5_2[_config][1], 10);
      break;
      */
      _subconfig = 1;
      if (_config == 0) 
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[0][1], 6);
        else 
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[0][1], 10);
          else
            add_triangle(tiling13_5_2[1][0], 10);
        }
      }
      else
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[1][1], 6);
        else 
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[1][1], 10);
          else
            add_triangle(tiling13_5_2[0][3], 10);
        }
      }
      break;
    case 25 :/* 13.5 */
      /*
      _subconfig = 2;
      if(test_interior(test13[_config][6]))
      add_triangle(tiling13_5_1[_config][2], 6);
      else
      add_triangle(tiling13_5_2[_config][2], 10);
      break;
      */
      _subconfig = 2;
      if(_config == 0)
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[0][2], 6);
        else 
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[0][2], 10);
          else
            add_triangle(tiling13_5_2[1][3], 10);
        }
      }
      else
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[1][2], 6);
        else
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[1][2], 10);
          else
            add_triangle(tiling13_5_2[0][0], 10);
        }

      }
      break;

    case 26 :/* 13.5 */
      /*
      _subconfig = 3;
      if(test_interior(test13[_config][6]))
      add_triangle(tiling13_5_1[_config][3], 6);
      else
      add_triangle(tiling13_5_2[_config][3], 10);
      break;
      */
      _subconfig = 3;
      if(_config == 0)
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[0][3], 6);
        else
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[0][3], 10);
          else
            add_triangle(tiling13_5_2[1][1], 10);
        }
      }
      else
      {
        if (interior_test_case13())
          add_triangle(tiling13_5_1[1][3], 6);
        else
        {
          if (tunnelOrientation == 1)
            add_triangle(tiling13_5_2[1][3], 10);
          else
            add_triangle(tiling13_5_2[0][2], 10);
        }
      }
      /* 13.4 common node is negative*/
      // v12 = add_c_vertex();
      // add_triangle(tiling13_4[_config][3], 12, v12);
      break;
    case 27 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][0], 10, v12);
      break;
    case 28 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][1], 10, v12);
      break;
    case 29 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][2], 10, v12);
      break;
    case 30 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][3], 10, v12);
      break;
    case 31 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][4], 10, v12);
      break;
    case 32 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][5], 10, v12);
      break;
    case 33 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][6], 10, v12);
      break;
    case 34 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][7], 10, v12);
      break;
    case 35 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][8], 10, v12);
      break;
    case 36 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][9], 10, v12);
      break;
    case 37 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][10], 10, v12);
      break;
    case 38 :/* 13.3 */
      v12 = add_c_vertex();
      add_triangle(tiling13_3_[_config][11], 10, v12);
      break;
    case 39 :/* 13.2 */
      add_triangle(tiling13_2_[_config][0], 6);
      break;
    case 40 :/* 13.2 */
      add_triangle(tiling13_2_[_config][1], 6);
      break;
    case 41 :/* 13.2 */
      add_triangle(tiling13_2_[_config][2], 6);
      break;
    case 42 :/* 13.2 */
      add_triangle(tiling13_2_[_config][3], 6);
      break;
    case 43 :/* 13.2 */
      add_triangle(tiling13_2_[_config][4], 6);
      break;
    case 44 :/* 13.2 */
      add_triangle(tiling13_2_[_config][5], 6);
      break;
    case 45 :/* 13.1 */
      add_triangle(tiling13_1_[_config], 4);
      break;

    default :
      printf("Marching Cubes: Impossible case 13?\n");
      print_cube();
    }
    break;

  case 14 :
    add_triangle(tiling14[_config], 4);
    break;
  };
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Adding triangles
void CMC33::add_triangle(const char* trig, char n, int v12)
  //-----------------------------------------------------------------------------
{
  int tv[3];

  for(int t = 0; t < 3*n; t++)
  {
    switch(trig[t])
    {
    case 0 :
      tv[ t % 3 ] = get_x_vert(_i , _j , _k);
      break;
    case 1 :
      tv[ t % 3 ] = get_y_vert(_i+1, _j , _k);
      break;
    case 2 :
      tv[ t % 3 ] = get_x_vert(_i ,_j+1, _k);
      break;
    case 3 :
      tv[ t % 3 ] = get_y_vert(_i , _j , _k);
      break;
    case 4 :
      tv[ t % 3 ] = get_x_vert(_i , _j ,_k+1);
      break;
    case 5 :
      tv[ t % 3 ] = get_y_vert(_i+1, _j ,_k+1);
      break;
    case 6 :
      tv[ t % 3 ] = get_x_vert(_i ,_j+1,_k+1);
      break;
    case 7 :
      tv[ t % 3 ] = get_y_vert(_i , _j ,_k+1);
      break;
    case 8 :
      tv[ t % 3 ] = get_z_vert(_i , _j , _k);
      break;
    case 9 :
      tv[ t % 3 ] = get_z_vert(_i+1, _j , _k);
      break;
    case 10 :
      tv[ t % 3 ] = get_z_vert(_i+1,_j+1, _k);
      break;
    case 11 :
      tv[ t % 3 ] = get_z_vert(_i ,_j+1, _k);
      break;
    case 12 :
      tv[ t % 3 ] = v12;
      break;
    default :
      break;
    }

    if(tv[t%3] == -1)
    {
      printf("Marching Cubes: invalid triangle %d\n", _ntrigs+1);
      print_cube();
    }

    if(t%3 == 2)
    {
      if(_ntrigs >= _Ntrigs)
      {
        m_MeshDataExt->m_Triangles.resize(2*_Ntrigs);
        _Ntrigs *= 2;
      }

      std::vector<BasicTriangle>::iterator tri = m_MeshDataExt->m_Triangles.begin() + _ntrigs++;

      tri->SetIndex(_ntrigs-1);
      tri->SetVert1Index(tv[0]);
      tri->SetVert2Index(tv[1]);
      tri->SetVert3Index(tv[2]);

      // Update the triangle membership list
      if (std::find(m_MeshDataExt->m_VertexToTriangleIndices[tv[0]].begin(),
                    m_MeshDataExt->m_VertexToTriangleIndices[tv[0]].end(),
                    _ntrigs-1)  == m_MeshDataExt->m_VertexToTriangleIndices[tv[0]].end())
      {
        m_MeshDataExt->m_VertexToTriangleIndices[tv[0]].push_back(_ntrigs-1);
      }

      if (std::find(m_MeshDataExt->m_VertexToTriangleIndices[tv[1]].begin(),
                    m_MeshDataExt->m_VertexToTriangleIndices[tv[1]].end(),
                    _ntrigs-1)  == m_MeshDataExt->m_VertexToTriangleIndices[tv[1]].end())
      {
        m_MeshDataExt->m_VertexToTriangleIndices[tv[1]].push_back(_ntrigs-1);
      }

      if (std::find(m_MeshDataExt->m_VertexToTriangleIndices[tv[2]].begin(),
                    m_MeshDataExt->m_VertexToTriangleIndices[tv[2]].end(),
                    _ntrigs-1)  == m_MeshDataExt->m_VertexToTriangleIndices[tv[2]].end())
      {
        m_MeshDataExt->m_VertexToTriangleIndices[tv[2]].push_back(_ntrigs-1);
      }


      // Update the vertex adjacency list
      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[0]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[0]].end(),
                    tv[1])  == m_MeshDataExt->m_VertexToVertexIndices[tv[0]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[0]].push_back(tv[1]);
      }

      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[0]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[0]].end(),
                    tv[2])  == m_MeshDataExt->m_VertexToVertexIndices[tv[0]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[0]].push_back(tv[2]);
      }


      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[1]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[1]].end(),
                    tv[0])  == m_MeshDataExt->m_VertexToVertexIndices[tv[1]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[1]].push_back(tv[0]);
      }

      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[1]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[1]].end(),
                    tv[2])  == m_MeshDataExt->m_VertexToVertexIndices[tv[1]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[1]].push_back(tv[2]);
      }


      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[2]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[2]].end(),
                    tv[0])  == m_MeshDataExt->m_VertexToVertexIndices[tv[2]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[2]].push_back(tv[0]);
      }

      if (std::find(m_MeshDataExt->m_VertexToVertexIndices[tv[2]].begin(),
                    m_MeshDataExt->m_VertexToVertexIndices[tv[2]].end(),
                    tv[1])  == m_MeshDataExt->m_VertexToVertexIndices[tv[2]].end())
      {
        m_MeshDataExt->m_VertexToVertexIndices[tv[2]].push_back(tv[1]);
      }

    }
  }
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// Calculating gradient

real CMC33::get_x_grad(const int i, const int j, const int k) const
  //-----------------------------------------------------------------------------
{
  if(i > 0)
  {
    if (i < _size_x - 1)
      return (get_data(i+1, j, k) - get_data(i-1, j, k)) / 2;
    else
      return get_data(i, j, k) - get_data(i-1, j, k);
  }
  else
    return get_data(i+1, j, k) - get_data(i, j, k);
}
//-----------------------------------------------------------------------------

real CMC33::get_y_grad(const int i, const int j, const int k) const
  //-----------------------------------------------------------------------------
{
  if(j > 0)
  {
    if (j < _size_y - 1)
      return (get_data(i, j+1, k) - get_data(i, j-1, k)) / 2;
    else
      return get_data(i, j, k) - get_data(i, j-1, k);
  }
  else
    return get_data(i, j+1, k) - get_data(i, j, k);
}
//-----------------------------------------------------------------------------

real CMC33::get_z_grad(const int i, const int j, const int k) const
  //-----------------------------------------------------------------------------
{
  if(k > 0)
  {
    if (k < _size_z - 1)
      return (get_data(i, j, k+1) - get_data(i, j, k-1)) / 2;
    else
      return get_data(i, j, k) - get_data(i, j, k-1);
  }
  else
    return get_data(i, j, k+1) - get_data(i, j, k);
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// Adding vertices

void CMC33::test_vertex_addition()
{
  if(_nverts >= _Nverts)
  {
    m_MeshDataExt->m_Vertices.resize(_Nverts*2);
    _Nverts *= 2;
  }
}


int CMC33::add_x_vertex()
  //-----------------------------------------------------------------------------
{
  test_vertex_addition();
  std::vector<BasicVertex>::iterator vert = m_MeshDataExt->m_Vertices.begin();
  vert += _nverts++;
  //BasicVertex *vert = _vertices + _nverts++;

  real u = (_cube[0]) / (_cube[0] - _cube[1]);

  vert->SetCoordX((real)_i+u);
  vert->SetCoordY((real) _j);
  vert->SetCoordZ((real) _k);

  if (_computeNormals)
  {
    vert->SetNormalX((1-u)*get_x_grad(_i,_j,_k) + u*get_x_grad(_i+1,_j,_k));
    vert->SetNormalY((1-u)*get_y_grad(_i,_j,_k) + u*get_y_grad(_i+1,_j,_k));
    vert->SetNormalZ((1-u)*get_z_grad(_i,_j,_k) + u*get_z_grad(_i+1,_j,_k));

    u = (real) sqrt(vert->GetNormalX() * vert->GetNormalX() 
                  + vert->GetNormalY() * vert->GetNormalY() 
                  + vert->GetNormalZ() * vert->GetNormalZ());
    if(u > 0)
    {
      vert->SetNormalX(vert->GetNormalX() / u);
      vert->SetNormalY(vert->GetNormalY() / u);
      vert->SetNormalZ(vert->GetNormalZ() / u);
    }
  }

  vert->SetIndex(_nverts-1);
  resizeAndAllocateConnectivity(_nverts-1);
  return _nverts-1;
}
//-----------------------------------------------------------------------------

int CMC33::add_y_vertex()
  //-----------------------------------------------------------------------------
{
  test_vertex_addition();
  std::vector<BasicVertex>::iterator vert = m_MeshDataExt->m_Vertices.begin();
  vert += _nverts++;

  real u = (_cube[0]) / (_cube[0] - _cube[3]);

  vert->SetCoordX((real) _i);
  vert->SetCoordY((real)_j+u);
  vert->SetCoordZ((real) _k);

  if (_computeNormals)
  {
    vert->SetNormalX((1-u)*get_x_grad(_i,_j,_k) + u*get_x_grad(_i,_j+1,_k));
    vert->SetNormalY((1-u)*get_y_grad(_i,_j,_k) + u*get_y_grad(_i,_j+1,_k));
    vert->SetNormalZ((1-u)*get_z_grad(_i,_j,_k) + u*get_z_grad(_i,_j+1,_k));

    u = (real) sqrt(vert->GetNormalX() * vert->GetNormalX()
                  + vert->GetNormalY() * vert->GetNormalY()
                  + vert->GetNormalZ() * vert->GetNormalZ());
    if(u > 0)
    {
      vert->SetNormalX(vert->GetNormalX() / u);
      vert->SetNormalY(vert->GetNormalY() / u);
      vert->SetNormalZ(vert->GetNormalZ() / u);
    }
  }

  vert->SetIndex(_nverts-1);
  resizeAndAllocateConnectivity(_nverts-1);
  return _nverts-1;
}
//-----------------------------------------------------------------------------

int CMC33::add_z_vertex()
  //-----------------------------------------------------------------------------
{
  test_vertex_addition();
  std::vector<BasicVertex>::iterator vert = m_MeshDataExt->m_Vertices.begin();
  vert += _nverts++;

  real u = (_cube[0]) / (_cube[0] - _cube[4]);

  vert->SetCoordX((real) _i);
  vert->SetCoordY((real) _j);
  vert->SetCoordZ((real)_k+u);

  if (_computeNormals)
  {
    vert->SetNormalX((1-u)*get_x_grad(_i,_j,_k) + u*get_x_grad(_i,_j,_k+1));
    vert->SetNormalY((1-u)*get_y_grad(_i,_j,_k) + u*get_y_grad(_i,_j,_k+1));
    vert->SetNormalZ((1-u)*get_z_grad(_i,_j,_k) + u*get_z_grad(_i,_j,_k+1));

    u = (real) sqrt(vert->GetNormalX() * vert->GetNormalX()
                  + vert->GetNormalY() * vert->GetNormalY()
                  + vert->GetNormalZ() * vert->GetNormalZ());
    if(u > 0)
    {
      vert->SetNormalX(vert->GetNormalX() / u);
      vert->SetNormalY(vert->GetNormalY() / u);
      vert->SetNormalZ(vert->GetNormalZ() / u);
    }
  }

  vert->SetIndex(_nverts-1);
  resizeAndAllocateConnectivity(_nverts-1);
  return _nverts-1;
}


int CMC33::add_c_vertex()
  //-----------------------------------------------------------------------------
{
  test_vertex_addition();
  std::vector<BasicVertex>::iterator vert = m_MeshDataExt->m_Vertices.begin() + _nverts++;

  real u = 0;
  int vid;

  if (_computeNormals)
  {
    vert->SetCoordX(0);
    vert->SetCoordY(0);
    vert->SetCoordZ(0);
    vert->SetNormalX(0);
    vert->SetNormalY(0);
    vert->SetNormalZ(0);
  }

  // Computes the average of the intersection points of the cube
  vid = get_x_vert(_i , _j , _k);
  if(vid != -1) 
  { 
    ++u; 
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_y_vert(_i+1, _j , _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());
    
    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_x_vert(_i ,_j+1, _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_y_vert(_i , _j , _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_x_vert(_i , _j ,_k+1);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_y_vert(_i+1, _j ,_k+1);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_x_vert(_i ,_j+1,_k+1);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_y_vert(_i , _j ,_k+1);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_z_vert(_i , _j , _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_z_vert(_i+1, _j , _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_z_vert(_i+1,_j+1, _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vid = get_z_vert(_i ,_j+1, _k);
  if(vid != -1)
  {
    ++u;
    const BasicVertex &v = m_MeshDataExt->m_Vertices[vid];
    vert->SetCoordX(vert->GetCoordX() + v.GetCoordX());
    vert->SetCoordY(vert->GetCoordY() + v.GetCoordY());
    vert->SetCoordZ(vert->GetCoordZ() + v.GetCoordZ());

    if (_computeNormals)
    {
      vert->SetNormalX(vert->GetNormalX() + v.GetNormalX());
      vert->SetNormalY(vert->GetNormalY() + v.GetNormalY());
      vert->SetNormalZ(vert->GetNormalZ() + v.GetNormalZ());
    }
  }

  vert->SetCoordX(vert->GetCoordX() / u);
  vert->SetCoordY(vert->GetCoordY() / u);
  vert->SetCoordZ(vert->GetCoordZ() / u);

  if (_computeNormals)
  {
    u = (real) sqrt(vert->GetNormalX() * vert->GetNormalX()
                  + vert->GetNormalY() * vert->GetNormalY()
                  + vert->GetNormalZ() * vert->GetNormalZ());
    if(u > 0)
    {
      vert->SetNormalX(vert->GetNormalX() / u);
      vert->SetNormalY(vert->GetNormalY() / u);
      vert->SetNormalZ(vert->GetNormalZ() / u);
    }
  }

  vert->SetIndex(_nverts-1);
  resizeAndAllocateConnectivity(_nverts-1);

  return _nverts-1;
}

void CMC33::resizeAndAllocateConnectivity(int index)
{
  size_t vert2TriSize = m_MeshDataExt->m_VertexToTriangleIndices.size()+1;

  if (index >= vert2TriSize)
  {
    // Resize the vector
    m_MeshDataExt->m_VertexToTriangleIndices.resize(vert2TriSize*2);

    // Assign an empty element
    vector<size_t> tri_indices;
    m_MeshDataExt->m_VertexToTriangleIndices[index] = tri_indices;
  }

  size_t vert2VertSize = m_MeshDataExt->m_VertexToVertexIndices.size()+1;
  if (index >= vert2VertSize)
  {
    // Resize the vector
    m_MeshDataExt->m_VertexToVertexIndices.resize(vert2VertSize*2);

    // Assign an empty element
    vector<size_t> vert_indices;
    m_MeshDataExt->m_VertexToVertexIndices[index] = vert_indices;
  }
}

//_____________________________________________________________________________

} // end of mitk namespace
//_____________________________________________________________________________
//_____________________________________________________________________________
