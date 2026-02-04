# Surgical Case Preparation Pipeline

## Overview
This repository contains an end-to-end pipeline for automating **surgical case preparation**, developed in Python during an internship at a MedTech startup and currently **used in production**.

The pipeline integrates **radiographic data, case spreadsheets, and 3D anatomical meshes** into a single, reproducible workflow for pre-operative planning.  
It replaces a previously manual process that took ~30 minutes per case with an automated pipeline that completes the same preparation in ~5 minutes, and has been used across **1,000+ real surgical cases**.

The core approach applies **singular value decomposition (SVD)** to surface-normal matrices to extract dominant geometric directions, enabling consistent, data-driven segmentation of surgical resection regions without case-by-case manual tuning.

---

## Motivation
Before this pipeline, surgical case preparation involved:
- manual alignment of anatomical meshes  
- hand-defined trimming planes  
- repeated visual inspection and adjustment  

This process was time-consuming, operator-dependent, and difficult to scale reliably.

The goal of this project was to:
- reduce manual labour and preparation time  
- improve consistency across cases  
- preserve geometric accuracy  
- support deployment in a real clinical planning workflow  

---

## Key Features
- **SVD-Based Segmentation**  
  Derived an algorithm to apply SVD to surface-normal matrices to identify principal geometric directions and define resection planes automatically.

- **End-to-End Automation**  
  Links Excel case metadata, radiographs, and 3D meshes into a single coordinated pipeline with minimal manual intervention.

- **Mesh Processing**  
  Includes trimming, filtering, and topological cleanup suitable for downstream CAD and simulation workflows.

- **Transformation Tracking**  
  Saves orientation, alignment, and trimming transforms to ensure reproducibility across patient datasets.

- **Production-Oriented Outputs**  
  Generates standardized output folders and logs designed for use by non-developers in a clinical setting.

---

## Technologies Used
- **Language:** Python  
- **Libraries:** NumPy, SciPy, Open3D, Trimesh, Pandas  
- **Visualization / QA:** CloudCompare, MeshLab  
- **Tooling:** Git, VS Code  

---

## Example Workflow
1. Import case metadata and associate radiographs with patient-specific 3D meshes  
2. Compute surface normals for relevant anatomical regions  
3. Apply SVD to extract dominant geometric directions  
4. Define and apply resection planes automatically  
5. Export cleaned meshes and transformation metadata for downstream use  

---

## Results
- Reduced surgical case preparation time from **~30 minutes to ~5 minutes per case**  
- Deployed in a production workflow and used across **1,000+ cases**  
- Improved consistency and reduced operator-dependent variability  
- Enabled scaling of surgical planning without proportional increases in manual labour  

---

## Notes
This repository focuses on the core geometric and systems pipeline.  
Company-specific integration details and patient data are not included.


---

## Acknowledgment
This repository contains work developed independently based on methods I designed during my internship at **Enhatch**, an AI-driven medtech company focused on surgical planning automation.  
All code and data here were created independently using open or synthetic sources, without proprietary content.
