# IGRSS Data Fusion Contest 2019 point cloud segmentation track

## Data description

Track 4 is the 3D Point Cloud Classification track. The goal is to classify (semantically segment) point clouds on a per point basis. The classes are:

| Class Index   | Class Description |
| :-----------: | ----------------- |
| 2             | Ground            |
| 5             | High Vegetation   |
| 6             | Building          |
| 9             | Water             |
| 17            | Bridge Deck       |

## Baseline

For the baseline algorithm, a [PointNet++](https://github.com/charlesq34/pointnet2) model was updated with modifications to support splitting/recombining large scenes.

## [PointSIFT](https://github.com/MVIG-SJTU/pointSIFT)
PointSIFT is a semantic segmentation framework for 3D point clouds. It is based on a simple module which extract featrues from neighbor points in eight directions.

## Usage
