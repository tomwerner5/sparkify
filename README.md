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

1. mini_sparkify_event_data.zip (the small, raw data file)
2. agg_large_sparkify.zip (the aggregated values from the full dataset)
3. Sparkify.ipynb (A Python Jupyter Notebook to be run locally)
4. run_sparkify.ipynb (A Python Jupyter Notebook to be run on AWS)
5. run_sparkify_offline.py (An alternative to running the model in AWS)

The full dataset is 12GB and is only provided here in aggregate form, but the full data can be obtained from the S3 bucket link provided in the .py file. The run_sparkify.ipynb file is intended to be executed using an AWS EMR/EC2 instance. A smaller anaysis (performed on a mini version of the dataset) can be executed from the Sparkify.ipynb notebook using the minified data (mini_sparkify_event_data.json).

## Some Notes on Setting up AWS

This [article](https://towardsdatascience.com/how-to-set-up-a-cost-effective-aws-emr-cluster-and-jupyter-notebooks-for-sparksql-552360ffd4bc) provides a relatively simple step-by-step process for setting up an AWS EMR instance. I followed this step by step, but I did run into a few issues along the way, which I'll outline here.

1. Unless you know what you're doing, I wouldn't use the bootstrap installation script. I only ran into errors running the author's script, and in the end I ditched it. Every time a cluster fails to boot up, AWS makes you either start over or clone the failed cluster settings and try again. There doesn't seem to be a way (or at least an easy way) to edit an existing cluster. Istead of using the boostrap script, simply run this line of code after initializing a spark session in your jupyter notebook to install packages: `SparkContext.install_pypi_package("package_you_want")`.
2. Do not use emr-5.30.0 as a cluster. Even though the article author recommends it, there are known issues with it, and I ran into those errors (see [this](https://stackoverflow.com/questions/61951352/notebooks-on-emr-aws-failed-to-start-kernel) as an example). Instead, I used emr-5.31.0. 
3. For the hardware configurations, I followed his exact instructions, except I wasn't allowed to have a value of '0' in the 'On-demand limit' field without also changing the core nodes to be spot pricing (which I ended up doing). 

My hardware setup consisted of 3 master nodes (each m5.xlarge), 3 core nodes (each m5.xlarge), and 3 task nodes (r4.2xlarge, r5.4xlarge, and r5.2xlarge). For the task and core nodes, I switched the memory to be HDD instead of SSD.

The total runtime of the notebook was about 2.5 hours, but debugging and setting up the cluster properly costed me about 20hrs of compute time. This ended up costing me about $30.

## Results <a name="results"></a>

The results can viewed in detail from the Jupyter Notebooks provided in this repository or in the medium article linked [here](somelink).

## Licensing and Acknowledgements<a name="licensing"></a>

The data was provided to me by Udacity. This is built under the MIT License.
