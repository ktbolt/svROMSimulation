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

import os
import json
import pdb
import numpy as np

from sv_rom_simulation.parameters import OutflowBoundaryConditionType


def outlet_bc_name(bc_type, face_name):
    """ The name an outlet's boundary condition is written under: its type and its face.

    The face rather than its position in the outlet list, because the name is what every
    result comes back labelled with -- svZeroDSolver's output, svZeroDVisualization, a 3D
    coupling block -- and 'RCR_14' says nothing about which vessel it is without the list
    beside it to look it up in.
    """
    return bc_type.upper() + '_' + face_name


def write_0d_solver_file(mesh, params, model):
    """
    Generate 0d solver input file (.json)
    """
    # create input dictionary
    inp = {'simulation_parameters': {},
           'boundary_conditions': [],
           'junctions': [],
           'vessels': []}

    # general
    inp['simulation_parameters']['model_name'] = params.model_name

    # time
    dt = params.time_step
    n_step = params.num_time_steps
    t_cycle = mesh.inflow_data[-1][0]
    check_flow_periods(mesh, t_cycle)
    # Rounded rather than truncated: a time step given as the period over a whole number of
    # points does not divide it exactly in floating point, and truncating 99.99999 lost a
    # point per cycle and, through the second line, a whole cycle.
    inp['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = int(round(t_cycle / dt))
    inp['simulation_parameters']['number_of_cardiac_cycles'] = int(round(n_step / t_cycle * dt)) + 1

    # fluid
    inp['simulation_parameters']['density'] = params.density
    inp['simulation_parameters']['viscosity'] = params.viscosity

    # vessels
    for (branch, ids), lengths in zip(mesh.cell_data['id'].items(), mesh.cell_data['length'].values()):
        for i, (j, length) in enumerate(zip(ids, lengths)):
            vessel = {'vessel_id': int(j),
                      'vessel_name': 'branch' + str(branch) + '_seg' + str(i),
                      'vessel_length': length,
                      'zero_d_element_type': 'BloodVessel',
                      'zero_d_element_values': {}}
            # zerod values
            for zerod in model['branches'].keys():
                vessel['zero_d_element_values'][zerod] = model['branches'][zerod][branch][i]
            # inlet bc
            if branch == 0 and i == 0:
                vessel['boundary_conditions'] = {'inlet': 'INFLOW'}
            # outlet bc
            if j in mesh.terminal:
                bc_name = list(mesh.outlet_face_names_index.keys())[mesh.terminal.index(j)]
                bc_str = outlet_bc_name(mesh.bc_type[bc_name], bc_name)
                vessel['boundary_conditions'] = {'outlet': bc_str}
            inp['vessels'] += [vessel]

    # junctions
    for i, j_branches in enumerate(mesh.seg_connectivity):
        # general junction properties
        junction = {'junction_name': 'J' + str(i),
                    'junction_type': 'NORMAL_JUNCTION',
                    'inlet_vessels': [int(j_branches[0])],
                    'outlet_vessels': []}
        for j in j_branches[1:]:
            junction['outlet_vessels'] += [int(j)]
        # branching vessels
        if len(j_branches) > 2:
            junction['tangents'] = mesh.junctions[i]['tangents']
            for n in ['areas', 'lengths']:
                junction[n] = [float(v) for v in mesh.junctions[i][n]]

        inp['junctions'] += [junction]

    # inlet bc
    inflow_q = []
    inflow_t = []
    for value in mesh.inflow_data:
        inflow_t += [value.time]
        inflow_q += [value.flow]
    inflow = {'bc_name': 'INFLOW',
              'bc_type': 'FLOW',
              'bc_values': {'t': inflow_t, 'Q': inflow_q}}
    inp['boundary_conditions'] += [inflow]

    # outlet bcs
    for bc_name, i in mesh.outlet_face_names_index.items():
        bc_type = mesh.bc_type[bc_name]
        bc_str = outlet_bc_name(bc_type, bc_name)
        bc_val = mesh.bc_map[bc_name]

        outflow = {'bc_name': bc_str,
                   'bc_type': bc_type.upper(),
                   'bc_values': {}}
        if bc_type == OutflowBoundaryConditionType.RCR:
            seq = ['Rp', 'C', 'Rd', 'Pd']
            for name, val in zip(seq, bc_val):
                outflow['bc_values'][name] = float(val)
        elif bc_type == OutflowBoundaryConditionType.RESISTANCE:
            seq = ['R', 'Pd']
            for name, val in zip(seq, bc_val):
                outflow['bc_values'][name] = float(val)
        elif bc_type == OutflowBoundaryConditionType.CORONARY:
            for name, val in bc_val['var'].items():
                outflow['bc_values'][name] = val
            outflow['bc_values']['t'] = bc_val['time']
            outflow['bc_values']['Pim'] = bc_val['pressure']
        elif bc_type == OutflowBoundaryConditionType.FLOW:
            # Negated: the file gives the flow into the model, and this condition is on a
            # vessel's outlet, where svZeroDSolver counts flow leaving the vessel as positive.
            # Written as it comes, an inflow would be drawn out of the model instead.
            outflow['bc_values']['t'] = [float(t) for t in bc_val['time']]
            outflow['bc_values']['Q'] = [-float(q) for q in bc_val['flow']]
        inp['boundary_conditions'] += [outflow]

    # write to file
    file_name = os.path.join(params.output_directory, params.solver_output_file)
    with open(file_name, 'w') as file:
        json.dump(inp, file, indent=4, sort_keys=True)


def check_flow_periods(mesh, t_cycle):
    """ Refuse prescribed flows whose period is not the inflow's.

    The cardiac cycle is read off the inflow file alone, so a second inflow over a different
    period would be sampled against the wrong cycle: svZeroDSolver repeats each waveform over
    its own last time, and two different periods drift apart cycle by cycle.
    """
    for name, bc_type in (mesh.bc_type or {}).items():
        if bc_type != OutflowBoundaryConditionType.FLOW:
            continue
        period = mesh.bc_map[name]['time'][-1]
        if not np.isclose(period, t_cycle):
            raise RuntimeError("The flow prescribed at '%s' has a period of %g s and the inflow one of %g s; "
                               "every prescribed flow has to cover the same cardiac cycle." % (name, period, t_cycle))
