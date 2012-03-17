/*=========================================================================

  Program:   Surface Registration Program
  Module:    $RCSfile: RegistrationMonitor.h,v $

  Copyright (c) Kitware Inc. 
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __RegistrationMonitor_h
#define __RegistrationMonitor_h

#include "itkMatrixOffsetTransformBase.h"
#include "itkCommand.h"
#include "itkOptimizer.h"

class vtkPolyData;
class vtkPolyDataMapper;
class vtkActor;
class vtkProperty;
class vtkMatrix4x4;
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;


/** \class RegistrationMonitor 
 *  This class provides a VTK visualization pipeline configured for monitoring
 *  the progress of a registration process. 
 */
class RegistrationMonitor
{
public:
  
  typedef RegistrationMonitor  Self;

  typedef itk::Optimizer       OptimizerType;

  typedef itk::MatrixOffsetTransformBase< double, 3, 3 > TransformType;

  RegistrationMonitor();
  ~RegistrationMonitor();

  void SetFixedSurface( vtkPolyData* surface );
  void SetMovingSurface( vtkPolyData* surface );

  void SetNumberOfIterationsPerUpdate( unsigned int number );
    
  void Observe( OptimizerType * optimizer, TransformType * transform );

  void SetVerbose( bool );

private:
  
  vtkMatrix4x4*                   Matrix;

  vtkPolyData*                    FixedSurface;
  vtkActor*                       FixedActor;
  vtkProperty*                    FixedProperty;
  vtkPolyDataMapper*              FixedMapper;

  vtkPolyData*                    MovingSurface;
  vtkActor*                       MovingActor;
  vtkProperty*                    MovingProperty;
  vtkPolyDataMapper*              MovingMapper;

  // Visualization pipeline
  vtkRenderer*                    Renderer;
  vtkRenderWindow*                RenderWindow;
  vtkRenderWindowInteractor*      RenderWindowInteractor;

  typedef itk::SimpleMemberCommand< Self >  ObserverType;

  ObserverType::Pointer           IterationObserver;
  ObserverType::Pointer           StartObserver;

  OptimizerType::Pointer          ObservedOptimizer;

  TransformType::Pointer          ObservedTransform;

  unsigned int                    CurrentIterationNumber;
  unsigned int                    NumberOfIterationsPerUpdate;
 
  bool                            Verbose;

  // These methods will only be called by the Observer
  void Update();
  void StartVisualization();

};

#endif
