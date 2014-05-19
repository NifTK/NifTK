#! /usr/bin/env python

import nipype.interfaces.utility  as util     # utility
import nipype.pipeline.engine     as pe          # pypeline engine
import nipype.interfaces.niftyseg as niftyseg
#import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.fsl      as fsl
import glob                       as filesearch
import os, re
from nipype.interfaces.base import InputMultiPath 
def create_niftyseg_gif_propagation_pipeline(name="niftyseg_gif_propagation", templatedatabase="/home/ntoussai/data/template-database", inputfile=''):
    """

    Creates a pipeline that uses seg_GIF label propagation to propagate 
    segmentation towards a target image
    
    Example
    -------

    >>> gif = create_niftyseg_gif_propagation_pipeline("niftyseg_gif")
    >>> gif.inputs.inputnode.input_image = 'T1.nii'
    >>> gif.run()                  # doctest: +SKIP

    Inputs::

        inputnode.input_image

    Outputs::


    """

    print "************************************************" 
    print "Looking for information in ", templatedatabase
    print "************************************************"
    
    T1s_directory = os.path.join(templatedatabase,"T1s")
    Cpps_directory = os.path.join(templatedatabase,"cpps")
    Labels_directory = os.path.join(templatedatabase,"labels")
    Template_db_file = os.path.join(templatedatabase,"db.xml")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name
    inputnode = pe.Node(interface = util.IdentityInterface(fields=["in_file", "db_file"]),
                        name="inputnode")
    inputnode.inputs.in_file = inputfile
    inputnode.inputs.db_file = Template_db_file
    bet = pe.Node(interface=fsl.BET(), name="bet")
    bet.inputs.mask = True

#    workflow.add_nodes([bet])
    workflow.connect([(inputnode, bet,[("in_file","in_file")])])
    
    gif = pe.Node(interface=niftyseg.Gif(), name="gif")
#    workflow.add_nodes([gif])    

    workflow.connect([(bet, gif,[("out_file","in_file")])])
    workflow.connect([(bet, gif,[("mask_file","mask_file")])])
    workflow.connect([(inputnode, gif,[("db_file","database_file")])])
    
    multiregistration_outputnode = pe.Node(interface = util.IdentityInterface(fields=["cpp_file", "cpp_dir"]),
                                           name="multiregistration")
    multiregistration_outputnode.inputs.cpp_dir = Cpps_directory

    labelfilelist=filesearch.glob(Labels_directory+"/*.nii.gz")  
    numberofimages = len(labelfilelist)
    print "Number of images found: \t%d" % numberofimages
    
    for floating in labelfilelist:
        
        floatingimage=os.path.basename(floating)
        floatingimagename=re.sub(r'\.nii.gz$', '', floatingimage)      
        print "Creating Registration node for ", floatingimagename
        nodename = ''.join(e for e in floatingimagename if e.isalnum())
#        regf3d = pe.Node(interface=niftyreg.Regf3d(), name="regf3d_"+floatingimagename)
        regf3d = pe.Node(interface=fsl.FNIRT(), name="regf3d_"+nodename)
#        workflow.add_nodes([regf3d])
        workflow.connect([(bet, regf3d,[("out_file","ref_file")])])
        floatingimagefile = os.path.join(T1s_directory, floatingimage)        
#        regf3d.inputs.flo_file = floatingimagefile
        regf3d.inputs.in_file = floatingimagefile
        cppfile = os.path.join(Cpps_directory, floatingimage)
#        regf3d.inputs.out_cpp_file = cppfile
        regf3d.inputs.field_file = cppfile
         
#        workflow.connect([(regf3d, multiregistration_outputnode,[("out_cpp_file","cpp_file")])])

    
    workflow.connect([(multiregistration_outputnode, gif,[("cpp_dir","cpp_dir")])])
    
    output_fields = ["out_dir"]
    outputnode = pe.Node(interface = util.IdentityInterface(fields=output_fields),
                         name="outputnode")

#    workflow.connect([(gif, outputnode, [("out_dir", "out_dir")])])

    return workflow
