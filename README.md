# Surgical Case Preparation Pipeline

## Overview
This repository contains an end-to-end pipeline for **automating surgical case preparation**, developed in Python.  
The system integrates **radiographic data, case spreadsheets, and 3D anatomical meshes** into a unified, reproducible workflow that streamlines pre-operative planning and segmentation.

The core algorithm applies **singular value decomposition (SVD)** on surface-normal matrices to isolate principal geometric directions, enabling data-driven segmentation of surgical resection regions.  
This replaces manual case-by-case processing with consistent, algorithmic methods.

---

## Key Features
- **SVD-Based Segmentation:** Applies SVD to surface normals to identify principal directions and segment surgical regions automatically.  
- **Data Integration:** Combines radiographs, Excel case data, and 3D mesh files into a single coordinated pipeline.  
- **Mesh Processing:** Includes trimming, filtering, and topological cleanup for downstream CAD or simulation use.  
- **Transformation Tracking:** Saves orientation and trimming matrices for reproducibility across patient datasets.  
- **Automated Reporting:** Generates standardized output folders and logs for each surgical case.

---

## Technologies Used
- **Languages:** Python  
- **Libraries:** NumPy, SciPy, Open3D, Trimesh, Pandas  
- **Applications:** CloudCompare, MeshLab, Excel  
- **Tools:** VS Code, GitHub

## Example Workflow
1. **Import case data** from Excel and link corresponding 3D meshes  
2. **Compute surface normals** for each anatomical region  
3. **Apply SVD** to extract principal components defining resection planes  
4. **Trim and isolate** the surgical region of interest  
5. **Export outputs** for visualization or integration into CAD workflows  

---

## Results
The pipeline transforms previously manual surgical planning tasks into reproducible algorithms, significantly reducing setup time and ensuring consistency across cases.  
Further iterations will incorporate curvature-based segmentation and semi-supervised learning for automatic region labeling.

---

## Acknowledgment
This repository contains work developed independently based on methods I designed during my internship at **Enhatch**, an AI-driven medtech company focused on surgical planning automation.  
All code and data here were created independently using open or synthetic sources, without proprietary content.
