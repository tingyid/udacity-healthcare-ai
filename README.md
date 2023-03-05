# Udacity - AI for Healthcare Nanodegree Program

<a href="https://confirm.udacity.com/XZUYPEH9">![ai4healthcare](certificate.png)</a>

In this nano-degree, I build, evaluate, and integrate predictive models that have the power to transform patient outcomes. I worked on

- classifying and segmenting 2D and 3D medical images to augment diagnosis;
- modeling patient outcomes with electronic health records to optimize clinical trial testing decisions;
- building an algorithm that uses data collected from wearable devices to estimate the wearer’s pulse rate in the presence of motion.

## Sections

### Applying AI to 2D & 3D Medical Imaging Data

In this section, I worked with both 2D & 3D medical imaging data and used AI to derive clinically-relevant insights from data gathered via different types of 2D medical images such as x-ray, mammography, and digital pathology, as well as 3D medical images such as MRI.

**High-level Medical Imaging AI Workflow:**

> 1. Extract images from DICOM files and apply the appropriate tools to perform exploratory data analysis.
> 2. Build AI models for different clinical scenarios that involve 2D images
> 3. Evaluate and position AI tools for regulatory approval.

**Projects**

**2D Image Processing: Pneumonia Detection from Chest X-Rays**
In this project,

> 1. I created a pipeline to extract images from DICOM files that can be fed into the CNN for model training;
> 2. analyzed data from the NIH Chest X-ray dataset and trained a CNN to classify a given chest X-ray for the presence or absence of pneumonia;
> 3. wrote an FDA 501(k) validation plan that formally describes my model, the data that it was trained on, and a validation plan that meets FDA criteria in order to obtain clearance of the software being used as a medical device.

**3D Image Processing: Hippocampus Volume Quantification for Alzheimer's Progression**

> Hippocampus is one of the major structures of the human brain with functions that are primarily connected to learning and memory. The volume of the hippocampus may change over time, with age, or as a result of disease. In order to measure hippocampal volume, a 3D imaging technique with good soft tissue contrast is required. MRI provides such imaging characteristics, but manual volume measurement still requires careful and time consuming delineation of the hippocampal boundary. In this project, you will go through the steps that will have you create an algorithm that will helps clinicians assess hippocampal volume in an automated way and integrate this algorithm into a clinician's working environment.

### Applying AI to EHR Data

In this section, I worked with EHR data to build and evaluate compliant and interpretable ML models. I analyzed EHR datasets, performed EDA and build powerful features with TensorFlow, and model the uncertainty and bias with TensorFlow Probability and Aequitas. I also researched on EHR data privacy and security standards.

**Project: Patient Selection for Diabetes Drug Testing**

> In this project, you are a data scientist for an exciting unicorn healthcare startup that has created a groundbreaking diabetes drug that is ready for clinical trial testing. You will work with real, de-identified EHR data to build a regression model to predict the estimated hospitalization time for a patient and select/filter patients for your study.

### Applying AI to Wearable Device Data

I worked with signal data collected by wearable device sensors such as IMU, PPG and ECG and built ML models to surface insights about the wearer’s health.

**Project: Motion Compensated Pulse Rate Estimation**

> Wearable devices have multiple sensors all collecting information about the same person at the same time. Combining these data streams allows us to accomplish many tasks that would be impossible from a single sensor. In this project, you will build an algorithm that combines information from two of the sensors that are covered in this course -- the IMU and PPG sensors -- to build an algorithm that can estimate the wearer’s pulse rate in the presence of motion. By only providing necessary background information and minimal starter code, this project puts you in the same position as a data scientist working for a wearable device company. You will have to rely on your knowledge of the sensors, the techniques that you have learned in this course, and your own creativity to design and implement an algorithm that accomplishes the task set out for you.
