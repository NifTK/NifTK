
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad                  $
 $Date:: 2011-09-20 14:34:44 +#$
 $Rev:: 7333                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <iostream>
using namespace std;

#include "ConversionUtils.h"
#include "CommandLineParser.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING, "message", "string", "Send a message to the world."},

  {OPT_DONE, NULL, NULL, 
   "What will you do for eternity?\n"
  }
};


enum { 
  O_MESSAGE
};


int main( int argc, char *argv[] )
{
  std::string message;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_MESSAGE, message );

  
  if (message.length() > 0)
    std::cout << message << std::endl;

  return EXIT_SUCCESS;     
}
