import os

from lark import Lark, inline_args, Transformer
import numpy as np


# This assumes that you're running experiments from the project root. Could be
# made more robust but a convenient assumption.
DATADIR = os.path.join(os.getcwd(), 'data')

### AMC
amc_grammar = r"""
  start: preamble keyframe*

  preamble: ":FULLY-SPECIFIED" ":DEGREES"

  keyframe: INT (measurement)+
  measurement: WORD " " (SIGNED_NUMBER " ")* SIGNED_NUMBER

  COMMENT: "#" /[^\n]/* /\n/

  %import common.INT
  %import common.NEWLINE
  %import common.SIGNED_NUMBER
  %import common.WORD

  %ignore COMMENT
  %ignore NEWLINE
"""

class AMCTreeToData(Transformer):
  def start(self, stuff):
    return stuff[1:]

  def keyframe(self, stuff):
    return int(stuff[0].value), dict(stuff[1:])

  def measurement(self, stuff):
    return stuff[0].value, [float(tok.value) for tok in stuff[1:]]

amc_parser = Lark(amc_grammar, parser='lalr', lexer='contextual')

### ASF
asf_preamble = """:version 1.10
:name VICON
:units
  mass 1.0
  length 0.45
  angle deg
:documentation
   .ast/.asf automatically generated from VICON data using
   VICON BodyBuilder and BodyLanguage model FoxedUp or BRILLIANT.MOD
"""

# For now we'll assume a fixed root position. It seems that all of the files
# follow this format, so we'll go ahead and enforce it.
asf_grammar = r"""
  start: "{preamble}" root bonedata hierarchy
  root: ":root" _NL root_order root_axis root_position root_orientation
    root_order: "order TX TY TZ RX RY RZ" _NL
    root_axis: "axis XYZ" _NL
    root_position: "position 0 0 0" _NL
    root_orientation: "orientation 0 0 0" _NL

  bonedata: ":bonedata" _NL joint*
  joint: _begin joint_id joint_name joint_direction joint_length joint_axis _joint_dof_clause? _end
    joint_id: "id" INT _NL
    joint_name: "name" WORD _NL
    joint_direction: "direction" SIGNED_NUMBER SIGNED_NUMBER SIGNED_NUMBER _NL
    joint_length: "length" SIGNED_NUMBER _NL
    joint_axis: "axis" SIGNED_NUMBER SIGNED_NUMBER SIGNED_NUMBER "XYZ" _NL
    _joint_dof_clause: joint_dof joint_limits
      joint_dof: "dof" (joint_dof_options)+ _NL
      !joint_dof_options: "rx" | "ry" | "rz"
      joint_limits: "limits" joint_limit+
      joint_limit: "(" SIGNED_NUMBER SIGNED_NUMBER ")" _NL

  hierarchy: ":hierarchy" _NL _begin hierarchy_parent_child* _end
  hierarchy_parent_child: WORD+ _NL

  _begin: "begin" _NL
  _end: "end" _NL

  COMMENT: "#" /[^\n]/* /\n/
  _NL: NEWLINE

  %import common.INT
  %import common.NEWLINE
  %import common.SIGNED_NUMBER
  %import common.WORD

  %ignore COMMENT

  %ignore " "
""".format(preamble=asf_preamble.replace('\n', '\\n'))

class ASFTreeToData(Transformer):
  def start(self, stuff):
    return {
      'root': stuff[0],
      'bonedata': stuff[1],
      'hierarchy': stuff[2]
    }

  def root(self, _):
    return {
      'order': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'],
      'position': [0, 0, 0],
      'orientation': [0, 0, 0]
    }

  def bonedata(self, stuff):
    return {joint['name']: joint for joint in stuff}

  def joint(self, stuff):
    return dict(stuff)

  def joint_id(self, stuff):
    return ('id', int(stuff[0].value))

  def joint_name(self, stuff):
    return ('name', stuff[0].value)

  def joint_direction(self, stuff):
    return ('direction', [float(stuff[i].value) for i in range(3)])

  def joint_length(self, stuff):
    return ('length', float(stuff[0].value))

  def joint_axis(self, stuff):
    return ('axis', [float(stuff[i].value) for i in range(3)])

  def joint_dof(self, stuff):
    return ('dof', stuff)

  def joint_dof_options(self, stuff):
    return stuff[0].value

  def joint_limits(self, stuff):
    return ('limits', stuff)

  def joint_limit(self, stuff):
    min_angle = float(stuff[0].value)
    max_angle = float(stuff[1].value)
    assert min_angle <= max_angle, 'min angle must be less than or equal to max angle'
    return (min_angle, max_angle)

  def hierarchy(self, stuff):
    parents = [p for p, _ in stuff]
    assert len(set(parents)) == len(parents), 'joint hierarchy contains a duplicate key'
    return dict(stuff)

  def hierarchy_parent_child(self, stuff):
    return (stuff[0].value, [s.value for s in stuff[1:]])

asf_parser = Lark(asf_grammar, parser='lalr', lexer='contextual')

def subject_trial_path(subject, trial, datadir=DATADIR):
  """Put together the path for the .amc file corresponding a particular trial by
  the given subject."""
  subject_str = '{:0>2d}'.format(subject)
  trial_str = '{:0>2d}'.format(trial)
  filename = '{}_{}.amc'.format(subject_str, trial_str)
  return os.path.join(
    datadir,
    'cmu_mocap',
    'subjects',
    subject_str,
    filename
  )

def subject_skeleton_path(subject, datadir=DATADIR):
  """Put together the path for the .asf file corresponding a particular
  subject."""
  subject_str = '{:0>2d}'.format(subject)
  filename = '{}.asf'.format(subject_str)
  return os.path.join(
    datadir,
    'cmu_mocap',
    'subjects',
    subject_str,
    filename
  )

def parse_amc_data(raw_contents):
  """Parse the string contents of an .amc file into a list of data items. Each
  list element is a pair consisting of the key frame id and a dict mapping joint
  names to lists of channels (rotation, etc)."""
  results = amc_parser.parse(raw_contents)
  return AMCTreeToData().transform(results)

def amc_to_array(parsed_amc, joint_order):
  """Take the parsed .amc file contents from `parse_amc_data()` and put them
  into an "array" (technically just a list of lists; you can choose what to do
  with it)."""
  # Check that all time points are in sorted order and accounted for
  assert [i for i, _ in parsed_amc] == list(range(1, len(parsed_amc) + 1))

  # Check that the joint names are the same across all time points and have the
  # same dimensionality
  joint_dims = {joint: len(parsed_amc[0][1][joint]) for joint in joint_order}
  assert all([
    sorted(joint_order) == sorted(kvs.keys())
    for _, kvs in parsed_amc
  ])
  assert all([
    len(vs) == joint_dims[k]
    for _, kvs in parsed_amc
    for k, vs in kvs.items()
  ])

  arr = [
    [val for jn in joint_order for val in kvs[jn]]
    for _, kvs in parsed_amc
  ]

  return joint_dims, arr

def load_skeleton(subject, datadir=DATADIR):
  path = subject_skeleton_path(subject, datadir=datadir)
  raw_contents = open(path, 'r').read()
  parsed = asf_parser.parse(raw_contents)
  return ASFTreeToData().transform(parsed)

def load_trial(subject, trial, joint_order=None, datadir=DATADIR):
  """Complete pipeline for loading subject/trial data into an array. The root
  joint is guaranteed to be the first in line for easier downstream processing.

  Returns
  =======
  joint_order : list of strings
  joint_dims : dict mapping joint names to its constituent number of channels
  arr : list of list of floats. The channel "array"
  """
  path = subject_trial_path(subject, trial, datadir=datadir)
  raw_contents = open(path, 'r').read()
  parsed = parse_amc_data(raw_contents)

  if joint_order is None:
    # Make sure to put root in front because we want to remove the first three
    # translation channels.
    joint_order = ['root'] + [k for k in parsed[0][1].keys() if k != 'root']

  joint_dims, arr = amc_to_array(parsed, joint_order)
  return joint_order, joint_dims, arr

def rotation_matrix(angles):
  """Construct a rotation matrix from Euler angles in radians."""
  [c1, c2, c3] = np.cos(angles)
  [s1, s2, s3] = np.sin(angles)

  rot_x = np.array(
    [[1,   0,  0],
     [0,  c1, s1],
     [0, -s1, c1]]
  )
  rot_y = np.array(
    [[c2, 0, -s2],
     [ 0, 1,   0],
     [s2, 0, c2]]
  )
  rot_z = np.array(
    [[ c3, s3, 0],
     [-s3, c3, 0],
     [  0,  0, 1]]
  )

  return rot_x @ rot_y @ rot_z

def frame_to_xyz(skeleton, frame):
  rot = {}
  xyz = {}

  # We are enforcing that the root dof are [tx, ty, tz, rx, ry, rz].
  rot['root'] = rotation_matrix(np.deg2rad(
    np.array(skeleton['root']['orientation']) +
    np.array(frame['root'][3:])
  ))
  xyz['root'] = (
    np.array(skeleton['root']['position']) +
    np.array(frame['root'][:3])
  )

  def visit(parent, joint):
    skel_joint = skeleton['bonedata'][joint]

    # The joint will not appear in the AMC file if it has not degrees of freedom
    dof = dict(zip(skel_joint.get('dof', []), frame.get(joint, [])))
    tdof = rotation_matrix(np.deg2rad(
      [dof.get('rx', 0), dof.get('ry', 0), dof.get('rz', 0)]
    ))
    torient = rotation_matrix(np.deg2rad(skel_joint['axis']))

    rot[joint] = np.linalg.solve(torient, tdof @ torient @ rot[parent])
    xyz[joint] = (
      xyz[parent] +
      skel_joint['length'] * np.array(skel_joint['direction']) @ rot[joint]
    )

    # Now visit all of our children. The joint will not show up in the hierarchy
    # if it has no children.
    for child in skeleton['hierarchy'].get(joint, []):
      visit(joint, child)

  # Start by visiting all of the children of the root
  for child in skeleton['hierarchy']['root']:
    visit('root', child)

  return xyz

def plot_skeleton(skeleton, xyz, axes=None):
  if axes is None:
    # For some reason this is required even though it's never even used.
    # pylint: disable=unused-import
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

  axes.set_aspect('equal')
  axes.invert_yaxis()

  # For some reason the xyz are out of order in the data.
  permutation = [0, 2, 1]
  xyz_stack = np.array(list(xyz.values()))[:, permutation]
  axes.scatter(xyz_stack[:, 0], xyz_stack[:, 1], xyz_stack[:, 2])
  for parent, children in skeleton['hierarchy'].items():
    for child in children:
      axes.plot(
        [xyz[parent][permutation[0]], xyz[child][permutation[0]]],
        [xyz[parent][permutation[1]], xyz[child][permutation[1]]],
        [xyz[parent][permutation[2]], xyz[child][permutation[2]]],
        color='grey'
      )

  # This is all one big hack to get around the fact that axis equal doesn't work
  # in mplot3d.
  mins = np.min(xyz_stack, axis=0)
  maxs = np.max(xyz_stack, axis=0)
  mids = 0.5 * (mins + maxs)
  radius = np.max(maxs - mins) * np.array([-0.5, 0.5])
  axes.auto_scale_xyz(mids[0] + radius, mids[1] + radius, mids[2] + radius)

  # Remove tick labels and grid lines.
  # ax.set_xticks([])
  # ax.set_yticks([])
  # ax.set_zticks([])

  # Remove tick labels, keep grid lines.
  axes.xaxis.set_ticklabels([])
  axes.yaxis.set_ticklabels([])
  axes.zaxis.set_ticklabels([])

# if __name__ == '__main__':
#   subject = 7
#   trial = 1
#   frame = 100

#   path = subject_skeleton_path(subject)
#   raw_contents = open(path, 'r').read()
#   parsed = asf_parser.parse(raw_contents)
#   skeleton = ASFTreeToData().transform(parsed)
#   frames = parse_amc_data(open(subject_trial_path(subject, trial), 'r').read())

#   xyz = frame_to_xyz(skeleton, frames[frame][1])
#   plot_skeleton(skeleton, xyz)
