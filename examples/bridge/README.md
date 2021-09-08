# bridge example

## This is a WIP - some things in this document do not yet exist.

## Overview

This example demonstrates how to set up a wobbling beam suspended between two
blocks. Key ideas include
- Setting up multiple meshes.
- Using tether forces at interfaces. The wobbling beam is tethered on its left
  and right to the blocks.
- Using tether forces to model structures that should not move (i.e., structures
  that are penalized from moving from their initial configuration)
- Moving from `IBAMR::IBFEMethod` to `fdl::IFEDMethod`.

In particular:
- bridge.cc is the same as IBFE/explicit/ex8.
- bridge_2.cc uses `IFEDMethod` for the beam and `IBFEMethod` for the blocks
  (perhaps the first time deal.II and libMesh have been used in the same code).
  This shows how to set up stress and boundary force functions with
  `IFEDMethod`. As usual, multiple `IBAMR::IBStrategy` objects can be combined
  with the `IBStrategySet` object.
- bridge_3.cc uses IFEDMethod for all three objects.

## how to set this up

Run
```shell
cmake -DFIDDLE_ROOT=path_to_fiddle .
make
```
where `path_to_fiddle` is either a build or installation directory of fiddle.
For example, if you built fiddle in `build/` in the top fiddle folder, then run
```shell
cmake -DFIDDLE_ROOT=$(pwd)/../../build/ .
make
```

to configure and build the example.
