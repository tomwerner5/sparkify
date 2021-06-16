# Sparkify Analysis

A user cancellation prediction model for the fictional Sparkify streaming platform.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

A basic installation of Python >= 3.6 is required. Besides the standard conda distribution, PySpark (and Apache Spark) will need to be installed. For a walkthrough on installing PySpark, see [here](https://www.datacamp.com/community/tutorials/installation-of-pyspark).

## Project Motivation<a name="motivation"></a>

This project is an exploration of PySpark and AWS. Proficiency in either of these could be a valuable skill set to have, so I set out to learn how to use them. 

## File Descriptions <a name="files"></a>

There are three files associated with this project, namely:

1. mini_sparkify_event_data.zip (the data file)
2. Sparkify.ipynb (A Python Jupyter Notebook)
3. run_sparkify.py

The full dataset is 12GB and not provided here, but can be obtained from the S3 bucket link provided in the .py file. The run_analysis.py file was intended to be executed using an AWS EMR instance. Details about the AWS instance can be found in the nedium [article](somelink). A smaller anaysis (performed on a mini version of the dataset) can be executed from the notebook with the data provided here.

## Results <a name="results"></a>

The results can viewed in detail from the Jupyter Notebook provided in this repository, or in summary form in the medium article linked [here](somelink).

## Licensing and Acknowledgements<a name="licensing"></a>

The data was provided to me by Udacity. This is built under the MIT License.
