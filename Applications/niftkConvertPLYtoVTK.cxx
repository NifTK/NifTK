/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>

#include <itkLogHelper.h>
#include <itkCommandLineHelper.h>
#include <niftkCommandLineParser.h>

#include <vtkPLYReader.h>
#include <vtkDebugLeaks.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRegressionTestImage.h>
#include <vtkTestUtilities.h>
#include <vtkPolyDataWriter.h>

#include <vtkWindowToImageFilter.h>

/*!
 * \file niftkConvertPLYtoVTK.cxx
 * \page niftkConvertPLYtoVTK
 * \section niftkConvertPLYtoVTKSummary Reads a Stanford University PLY polygonal file and converts it to VTK format.
 *
 * \section niftkConvertPLYtoVTKCaveats Caveats
 * \li None
 */

struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  bool flgRender;

  std::string fileInputImage;
  std::string fileOutputImage;
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
  }
};


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_SWITCH, "r", 0,   "Open a VTK render window and display the surface."},

  {OPT_STRING|OPT_REQ, "o", "filename", "The output file."},
  {OPT_STRING|OPT_REQ, "i", "filename", "The input PLY file."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to convert a Stanford University PLY polygonal file format and convert it to VTK format.\n"
  }
};

enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_RENDER,

  O_OUTPUT_IMAGE,
  O_INPUT_IMAGE
};


/**
 * \brief Performs a image moments  registration
 */
int main(int argc, char** argv)
{

  // To pass around command line args
  struct arguments args;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);


  CommandLineOptions.GetArgument( O_DEBUG,   args.flgDebug );
  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_RENDER, args.flgRender );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.fileOutputImage );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE,  args.fileInputImage );


  // Test if the reader thinks it can open the file.
  int canRead = vtkPLYReader::CanReadFile( args.fileInputImage.c_str() );
  (void)canRead;

  // Create the reader.
  vtkPLYReader* reader = vtkPLYReader::New();
  reader->SetFileName( args.fileInputImage.c_str() );
  reader->Update();

  // Write the data
  vtkPolyDataWriter* writer = vtkPolyDataWriter::New();

  writer->SetFileName( args.fileOutputImage.c_str() );
  writer->SetInputConnection( reader->GetOutputPort() );
  writer->Write();

  if ( args.flgRender )
  {

    // Create a mapper.
    vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->SetInputConnection(reader->GetOutputPort());
    mapper->ScalarVisibilityOn();
    
    // Create the actor.
    vtkActor* actor = vtkActor::New();
    actor->SetMapper(mapper);
    
    // Basic visualisation.
    vtkRenderWindow* renWin = vtkRenderWindow::New();
    vtkRenderer* ren = vtkRenderer::New();
    renWin->AddRenderer(ren);
    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    iren->SetRenderWindow(renWin);
    
    ren->AddActor(actor);
    ren->SetBackground(0,0,0);
    renWin->SetSize(300,300);
    
    // interact with data
    renWin->Render();

    iren->Start();

    actor->Delete();
    mapper->Delete();
    renWin->Delete();
    ren->Delete();
    iren->Delete();
  }

  reader->Delete();
  writer->Delete();

  return EXIT_SUCCESS;
}
