# Notes

## Big PIcture

- Want to be able to convert this new network into a system that can accept our data and produce a reasonable output
    - we want this bc it will improve our network capabilities

## What is in our way?
- System is really complex
- lots of moving parts, lots of linkages between files
- dependencies out the wazoo
- Proprietary (not REALLY) configuration system (it has to be their way, not our way)

## Specific Impediments
- Don't know how to parse out a single model and backbone and make them work together
    - want this to be self-contained (not split across a bunch of files)

## My thought process
- Find model and backbone that I want
- Line by line through the file and create a sepearte directory with all dependencies

## Extra Notes
- SegFix is a "Net" that you run after HRnet to fix your segmentation. (the 2nd step in a 2 step process) - run it after you ahve trained the network

# Resources
SegFix:
https://arxiv.org/pdf/2007.04269.pdf

HRNet:
https://arxiv.org/pdf/1908.07919.pdf