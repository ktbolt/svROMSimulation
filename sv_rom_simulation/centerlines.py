# Copyright (c) Stanford University, The Regents of the University of
#               California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
The module is used to extact centerlines from a surface mesh. 
"""

import os
from collections import namedtuple
from pathlib import Path
import numpy as np

import logging
from sv_rom_simulation.manage import get_logger_name

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vmtk import vtkvmtk

from sv_rom_simulation.mesh import outlet_end_points
from sv_rom_simulation.utils import SurfaceFileFormats, read_surface, read_polydata, write_polydata

class Centerlines(object):
    """
    The Centerlines class is used to encapsulate centerline calculations.
    """
    def __init__(self):

        self.inlet_face_name = None
        self.inlet_center = None
        self.outlet_centers = None
        self.outlet_face_names = None
        self.geometry = None
        self.cap_ids = None

        self.logger = logging.getLogger(get_logger_name())

    def compute_branch_splitting_centerlines(self, params):
        """
        Compute the centerlines for a closed surface.
        The centerline geometry is returned as a vtkPolyData object.
        """
        model_surface = read_surface(params.surface_model)
        caps = get_caps(model_surface, params.boundary_surfaces_dir)
        inlet_face_id = get_inlet_face_id(params, caps)
        face_centers = {face_id: cap.center for face_id, cap in caps.items()}
        self.logger.info("Inlet face %d, %d outlet faces: %s" % (
            inlet_face_id, len(caps) - 1,
            ", ".join(cap.name or str(face_id) for face_id, cap in caps.items() if face_id != inlet_face_id)))

        self.logger.info("Computing surface centerlines ...")
        centerlines = self.compute_centerlines(model_surface, face_centers, inlet_face_id)
        self.logger.info("The surface centerlines have been computed.")

        self.logger.info("Computing branch splitting centerlines ...")
        branch_splitting = vtkvmtk.vtkvmtkPolyDataCenterlineBranchSplitting()
        branch_splitting.SetInputData(model_surface)
        branch_splitting.SetCenterlines(centerlines)

        branch_splitting.SetCenterlineSectionAreaArrayName('CenterlineSectionArea')
        branch_splitting.SetCenterlineSectionMinSizeArrayName('CenterlineSectionMinSize')
        branch_splitting.SetCenterlineSectionMaxSizeArrayName('CenterlineSectionMaxSize')
        branch_splitting.SetCenterlineSectionShapeArrayName('CenterlineSectionShape')
        branch_splitting.SetCenterlineSectionClosedArrayName('CenterlineSectionClosed')
        branch_splitting.Update()
        branch_centerlines = branch_splitting.GetCenterlines()
        self.logger.info("The branch splitting centerlines have been computed.")
        self.geometry = branch_centerlines
        self.name_outlets(caps, inlet_face_id)

        write_polydata(params.centerlines_output_file, self.geometry)

        '''
        # Get the centers of the inlet and outlet surfaces.
        self.get_inlet_outlet_centers(params)

        # Read the surface model used for centerline calculation.
        self.logger.info("Read surface model from %s" % params.surface_model)
        self.logger.info("Number of points in params.outlet_centers %d" % len(self.outlet_centers))

        # Extract centerlines using SimVascular.
        self.logger.info("Calculating surface centerlines ...")

        params_sv = {'surf_in': params.surface_model,
                     'caps': self.cap_ids,
                     'cent_out': params.centerlines_output_file}
        sv_centerlines(params_sv)

        self.geometry = read_surface(params.centerlines_output_file)
        self.logger.info("The surface centerlines have been calculated.")

        # Write outlet face names.
        self.write_outlet_face_names(params)
        '''

    def compute_centerlines(self, surface, face_centers, inlet_face_id):
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(surface)
        pointLocator.BuildLocator()

        source_ids = vtk.vtkIdList()
        target_ids = vtk.vtkIdList()

        for id,center in face_centers.items():
            pt_id = pointLocator.FindClosestPoint(center)
            if id == inlet_face_id:
                source_ids.InsertNextId(pt_id)
            else:
                target_ids.InsertNextId(pt_id)

        centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(surface)
        centerlineFilter.SetSourceSeedIds(source_ids)
        centerlineFilter.SetTargetSeedIds(target_ids)

        centerlineFilter.SetRadiusArrayName("MaximumInscribedSphereRadius")
        centerlineFilter.SetCostFunction("1/R")
        centerlineFilter.SetFlipNormals(False)
        centerlineFilter.SetAppendEndPointsToCenterlines(True)
        centerlineFilter.SetSimplifyVoronoi(False)
        centerlineFilter.SetResamplingStepLength(True)
        centerlineFilter.Update()
        centerlines = centerlineFilter.GetOutput()

        return centerlines

    def name_outlets(self, caps, inlet_face_id):
        """
        Name each outlet end of the centerlines after the cap it ends at.

        Paired by position, one to one and closest first, rather than by the order the caps
        were handed to VMTK in. The outlet names are matched to the boundary conditions by
        list position (Mesh.terminal), and nothing ties the order the centerline ends come out
        in to the order their seeds went in: a list that is off by one names every boundary
        condition after its neighbour's vessel, and nothing downstream can tell -- the ids
        are all valid and the names all real. Where a centerline end is is not open to that.

        Left unnamed (and so to the outlet face names file, in centerline order, as before)
        where the caps came without names.
        """
        outlets = {face_id: cap for face_id, cap in caps.items() if face_id != inlet_face_id}
        ends = outlet_end_points(self.geometry)
        if len(ends) != len(outlets):
            raise RuntimeError("The centerlines have %d outlet ends and the surface %d outlet caps: a centerline "
                               "did not reach every cap, or ran out through something that is not one." %
                               (len(ends), len(outlets)))
        if any(cap.name is None for cap in outlets.values()):
            return

        from scipy.optimize import linear_sum_assignment
        face_ids = list(outlets)
        end_points = np.array([self.geometry.GetPoint(int(point)) for _branch, point in ends])
        centers = np.array([outlets[face_id].center for face_id in face_ids])
        distances = np.linalg.norm(end_points[:, None, :] - centers[None, :, :], axis=-1)
        rows, columns = linear_sum_assignment(distances)

        names = [None] * len(ends)
        for row, column in zip(rows, columns):
            cap = outlets[face_ids[column]]
            # A centerline ends on the surface point nearest its cap's center, so within the
            # cap; one a cap's width away ended somewhere else and was paired by elimination.
            if distances[row, column] > cap.diameter:
                raise RuntimeError("The centerline end of branch %d is %.3g from the center of '%s', the nearest "
                                   "cap left to pair it with, which is %.3g across." %
                                   (ends[row][0], distances[row, column], cap.name, cap.diameter))
            names[row] = cap.name
        self.outlet_face_names = names
        self.logger.info("Outlets in centerline order: %s" % ", ".join(names))

    def read(self, file_name):
        """
        Read centerlines from a .vtp file.
        """
        self.geometry = read_polydata(file_name)


    def get_inlet_outlet_centers(self, params):
        """
        Get the centers of the inlet and outlet surface geometry.
        Surface inlet and outlet faces are identifed by their file name.
        """
        surface = read_polydata(params.surface_model)
        faces = get_surface_faces(surface)
        self.logger.info("Number of surface faces: %d" % len(faces))
        face_centers = get_face_centers(faces)

        #inlet_face_id = params.inlet_face_id
        #self.inlet_center = get_polydata_centroid(polydata)

        '''
        surf_mesh_dir = Path(params.boundary_surfaces_dir)
        inlet_file_name = params.inlet_face_input_file
        self.outlet_face_names = []
        self.outlet_centers = []

        for face_file in surf_mesh_dir.iterdir():
            file_name = face_file.name
            self.logger.debug("Surface file name: %s" % file_name)
            file_suffix = face_file.suffix.lower()[1:]

            if file_suffix not in SurfaceFileFormats or file_name.lower().startswith('wall'):
                continue

            if file_name == inlet_file_name:
                inlet_path = str(face_file.absolute())
                self.logger.info("Inlet file: %s" % inlet_path)
                polydata = read_surface(inlet_path, file_suffix)
                self.inlet_center = get_polydata_centroid(polydata)
                self.inlet_face_name = face_file.stem 
            else:
                outlet_path = str(face_file.absolute())
                self.outlet_face_names.append(face_file.stem)
                self.logger.info("Outlet: %s" % file_name)
                polydata = read_surface(outlet_path, file_suffix)
                # Use extend because vmtk expects a list of floats.
                self.outlet_centers.extend(get_polydata_centroid(polydata))

        if not self.inlet_face_name:
            raise RuntimeError("No inlet face found in the boundary surface directory '%s'" % params.boundary_surfaces_dir)

        # get surface points closest to cap centers
        cp = ClosestPoints(read_surface(params.surface_model))
        caps = np.vstack((self.inlet_center, np.array(self.outlet_centers).reshape(-1, 3)))
        self.cap_ids = cp.search(caps)
        '''

        self.logger.info("Number of outlet faces: %d" % len(self.outlet_centers))

    def write_outlet_face_names(self, params):
        """
        Write outlet face names
        """
        file_name = os.path.join(params.output_directory, params.CENTERLINES_OUTLET_FILE_NAME)
        self.logger.info("Write outlet face names to: %s" % file_name) 
        with open(file_name, "w") as fp:
            for name in self.outlet_face_names:
                fp.write(name+"\n")


# A cap: the face id it carries on the surface, the name it goes by (None where only the
# geometry said it was a cap), its center, and the diameter of the circle of its area.
Cap = namedtuple('Cap', 'name center diameter')

# Largest distance of a face from its own best-fit plane, as a fraction of the face's size,
# below which the face is taken for a cap. A cap cut across a vessel and meshed comes out at
# a few hundredths -- the mesher moves its interior points off the cut by a fraction of an
# element -- and a wall, curving round the vessel, at a large fraction of its size.
CAP_FLATNESS_TOLERANCE = 0.1


def get_caps(surface, boundary_surfaces_dir=None):
    '''The surface's caps by face id, named where the face files say what they are called.

    The face files are what a mesh-complete folder's mesh-surfaces directory holds, and their
    names are the SimVascular convention the rest of this package follows: a file whose name
    starts with 'wall' is wall, anything else a cap, and the file's stem the face's name. That
    is a statement about every face, so it is taken over the geometry wherever it is there.
    Without them the caps are the faces that are flat.
    '''
    caps = caps_from_face_files(surface, boundary_surfaces_dir) if boundary_surfaces_dir else {}
    if not caps:
        caps = {face_id: Cap(None, get_polydata_centroid(face), equivalent_diameter(face))
                for face_id, face in get_surface_faces(surface).items()}
    if len(caps) < 2:
        raise RuntimeError("Found %d cap(s) on the surface, and centerlines need an inlet and at least one "
                           "outlet." % len(caps))
    return caps


def caps_from_face_files(surface, boundary_surfaces_dir):
    '''The caps among the face files in a directory: {face id: Cap}, or {} if it holds none.

    Only files carrying a single ModelFaceID that the surface also carries count, so that a
    directory holding something else -- the default is the working directory -- reads as
    holding no faces rather than as faces of this surface.
    '''
    surface_ids = set(np.unique(v2n(surface.GetCellData().GetArray('ModelFaceID'))).tolist())
    caps = {}
    for face_file in sorted(Path(boundary_surfaces_dir).iterdir()):
        suffix = face_file.suffix.lower()[1:]
        if suffix not in SurfaceFileFormats or face_file.name.lower().startswith('wall'):
            continue
        face = read_surface(str(face_file), suffix)
        ids = face.GetCellData().GetArray('ModelFaceID')
        if ids is None or face.GetNumberOfCells() == 0:
            continue
        face_ids = np.unique(v2n(ids)).tolist()
        if len(face_ids) != 1 or face_ids[0] not in surface_ids:
            continue
        caps[int(face_ids[0])] = Cap(face_file.stem, get_polydata_centroid(face), equivalent_diameter(face))
    return dict(sorted(caps.items()))


def get_inlet_face_id(params, caps):
    '''The inlet's face id: as given, or the id of the cap the inlet face file names.'''
    if params.inlet_face_id is not None:
        inlet_face_id = int(params.inlet_face_id)
    elif params.inlet_face_input_file:
        stem = Path(params.inlet_face_input_file).stem
        named = [face_id for face_id, cap in caps.items() if cap.name == stem]
        if not named:
            raise RuntimeError("The inlet face '%s' is not among the caps: %s." %
                               (stem, ", ".join(cap.name or str(face_id) for face_id, cap in caps.items())))
        inlet_face_id = named[0]
    else:
        raise RuntimeError("No inlet face: give its face id or its face file.")
    if inlet_face_id not in caps:
        raise RuntimeError("Face %d is not a cap of the surface, so it cannot be the inlet. The caps are %s." %
                           (inlet_face_id, ", ".join(str(face_id) for face_id in caps)))
    return inlet_face_id


def equivalent_diameter(face):
    '''Diameter of the circle of the face's area.'''
    properties = vtk.vtkMassProperties()
    properties.SetInputData(face)
    properties.Update()
    return 2.0 * float(np.sqrt(properties.GetSurfaceArea() / np.pi))


def get_surface_faces(surface):
    '''Get the flat faces from the surface mesh using the ModelFaceID data array.
    '''
    face_ids = v2n(surface.GetCellData().GetArray('ModelFaceID'))

    ## Extract face geometry.
    #
    # Over the ids the surface carries rather than every integer between the smallest and the
    # largest: a gap in the numbering is an empty face, which has no plane to be flat in.
    faces = {}
    for i in np.unique(face_ids).tolist():
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(surface)
        threshold.SetInputArrayToProcess(0,0,0,1,'ModelFaceID')
        threshold.SetLowerThreshold(i)
        threshold.SetUpperThreshold(i)
        threshold.Update();

        surfacer = vtk.vtkDataSetSurfaceFilter()
        surfacer.SetInputData(threshold.GetOutput())
        surfacer.Update()
        face = surfacer.GetOutput()

        if surface_is_flat(face):
            faces[int(i)] = face

    return faces

def get_face_centers(model_faces):
    face_centers = {}

    for id,face in model_faces.items():
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(face)
        centerOfMassFilter.SetUseScalarsAsWeights(False)  
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()
        face_centers[id] = center

    return face_centers

def surface_is_flat(surface):
    '''Whether a face lies in a plane, to within CAP_FLATNESS_TOLERANCE of its own size.

    Measured as distance from the best-fit plane rather than as Gaussian curvature. The
    curvature of a meshed cap is not zero: its interior points sit a fraction of an element off
    the cut, which reads as curvature of order 1e-3 per square unit at the smallest, and an
    absolute tolerance of 1e-5 found no caps at all on a clinical mesh -- so no centerline
    targets, and centerlines with no cells. A distance relative to the face's size does not
    depend on the units or the resolution either.
    '''
    if surface.GetNumberOfPoints() < 3:
        return False
    points = v2n(surface.GetPoints().GetData()).astype(float)
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    deviation = np.abs((points - centroid) @ normal).max()
    size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return bool(deviation <= CAP_FLATNESS_TOLERANCE * size)





def get_polydata_centroid(poly_data):
    """
    Calculate the centroid of polydata
    """
    return np.mean(v2n(poly_data.GetPoints().GetData()), axis=0).tolist()


def sv_centerlines(p):
    """
    Call SimVascular centerline generation
    """
    try:
        import sv
    except ImportError:
        raise ImportError('Run with sv --python -- this_script.py')

    # create a modeler
    kernel = sv.modeling.Kernel.POLYDATA
    modeler = sv.modeling.Modeler(kernel)

    # read surface mesh
    model = modeler.read(p['surf_in'])
    model_polydata = model.get_polydata()

    # generate centerline
    # todo: try use_face_ids
    centerlines_polydata = sv.vmtk.centerlines(model_polydata, [p['caps'][0]], p['caps'][1:], use_face_ids=False)

    # write centerline to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(p['cent_out'])
    writer.SetInputData(centerlines_polydata)
    writer.Update()
    writer.Write()


class ClosestPoints:
    """
    Find closest points within a geometry
    """
    def __init__(self, inp):
        dataset = vtk.vtkPolyData()
        dataset.SetPoints(inp.GetPoints())

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        self.locator = locator

    def search(self, points):
        """
        Get ids of points in geometry closest to input points
        Args:
            points: list of points to be searched

        Returns:
            Id list
        """
        ids = []
        for p in points:
            ids += [self.locator.FindClosestPoint(p)]
        return ids
