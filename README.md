# README for DAC \(TimingPredict\) Fork Project

# Project Overview

This project is a fork of the original DAC \(TimingPredict\) repository, developed for academic research purposes\. We have made significant optimizations and extensions based on the original codebase to enhance its performance, stability, and applicability in timing prediction tasks for VLSI circuits\.

# Key Optimizations \&amp; Extensions

Building upon the original TimingPredict, our main improvements and additions are as follows:

- **Model \&amp; Data Preprocessing Optimization**: We optimized the original model structure and data preprocessing pipeline to improve prediction accuracy and efficiency, addressing potential bottlenecks in feature extraction and model inference\.

- **Training Stability Analysis**: Conducted in\-depth analysis on the stability of model training, identifying factors that affect training convergence and proposing corresponding optimization strategies to ensure consistent and reliable training results\.

- Custom Dataset Construction Across Multiple Process Corners: Independently constructed datasets covering more process corners, adopting a more robust training method to enhance the model\&\#39;s adaptability to different process variations\.

- **Cross\-Corner Generalization Testing**: Evaluated the model\&\#39;s cross\-corner generalization capability, verifying its performance on unseen process corners to demonstrate its practical value in real\-world VLSI design scenarios\.

- **Visualization with OpenROAD**: Integrated the open\-source tool OpenROAD to realize visualization of critical paths and violation paths, facilitating intuitive analysis of timing prediction results and circuit performance bottlenecks\.

# Reproducibility Support

To facilitate easy reproduction of our experimental results, we provide the required graph file for model inference of a circuit under the skywater130 ff process corner\. The graph file is stored in Baidu Netdisk, and the download link is provided below:

**Baidu Netdisk Download Link**: \[Insert Your Baidu Netdisk Link Here\]

Note: This graph file is essential for running the model inference on the skywater130 ff circuit, ensuring consistency with our experimental setup\.

# Environment Requirements

## Basic Environment \(Inherited from TimingPredict\)

For the basic environment, please refer to the original TimingPredict repository\&\#39;s requirements\. Ensure all dependencies specified in the original project are installed correctly before proceeding with our extensions\.

## Additional Environment for OpenSTA/OpenROAD

If you need to extract features independently, you will need an additional set of environments for OpenSTA and OpenROAD\. Please refer to the official documentation of [OpenROAD](https://www.opensroad.io/) and [OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) for detailed installation instructions\.

## DGL Environment Notes

Currently, the Deep Graph Library \(DGL\) used in this project lacks maintenance support for Windows\. There are two options to use DGL in your environment:

1. **Linux Environment**: DGL can be installed directly via pip following the official guide, which is the recommended approach for stability and simplicity\.

2. **Windows Environment**: You can build DGL from source, though this process is relatively complex\. To simplify this, we provide our precompiled DGL package for Windows users who want to quickly complete testing\. Our precompiled package is compatible with the following environment:

    - Python 3\.12

    - PyTorch 2\.5

    - CUDA 12\.1

Note: The environment for building DGL from source is: Windows 11 \(x86\), Python 3\.12, CUDA 12\.8\. If you choose to build from source, please ensure your environment matches these specifications\.

# Usage Guide

1. Clone this repository to your local machine\.

2. Install the basic environment following the original TimingPredict guidelines\.

3. Install additional environments for OpenSTA/OpenROAD if feature extraction is needed\.

4. Download the provided graph file from Baidu Netdisk and place it in the specified directory \(see project structure for details\)\.

5. Run the model inference or training scripts according to your research needs\. Refer to the original TimingPredict documentation for basic usage, and our extended scripts for new features \(e\.g\., visualization, cross\-corner testing\)\.

# Original Repository

This project is a fork of the original DAC \(TimingPredict\) repository\. For more details on the base code, please refer to the original repository: \[Insert Original TimingPredict Repository Link Here\]

# Contact

For any questions, issues, or suggestions regarding this project, please feel free to contact us via \[Insert Your Contact Information Here\]\.

> （注：文档部分内容可能由 AI 生成）
