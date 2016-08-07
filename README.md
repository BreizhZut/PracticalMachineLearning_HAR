# Practical Machine Learning: Human Activity Recognition Assignment

Coursera Data Science Specialisation Assignment: Human Activity Recognition, machine learning. 

## Overview

This repository is the completion of the Practical Machine Learning Assignment of the Data Science specialisation provided by John Hopkins University on Coursera.

## Background (copy from the Assigment web page)

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har] (see the section on the Weight Lifting Exercise Dataset).


## Assignent objectives

We are provided with 2 data set:

* `HARtrain.csv` is a very large data set, containing the exercise classification (how well de barbell lifts were performed)
*  `HARtest.csv` constitute a blind test (for which we don't know how well the exercice was perforemed) of 20 observations. 
 
 We use the first data set to create a model based on machine learning algorithms. Data analysis is to be described in a (<2000 word document). 
 The best  model is then to be applied to the second (blind) data set.
 
 Evaluation is performed as:
 
 1. Peer review of the model documentation
 2. Automated quiz to evaluate the performance of the model on the blind test.

## Content

As requested for the Assignment documentation described the analysis is uploaded in this github repository. 
These files describe the analysis and practical machine learning implementation used for this project. Since from experience the html format can't be large and may not display in github, we also provide the markdown version after making uploaded the relevant figure.

 	 this `README`, this repository contains the following files

* Markdown document, `HAR_quality.md`
* Figure within directory `HAR_quality_files/figures_html`
* Corresponding html document `HAR_quality.html`

This assignment was performed with the `knitr` package for R (version 3.2.4) using Rstudio (Version 0.99.892) on a MacBookPro (2.9 GHz Intel Core i7) running on Mac OS X 10.11.6 (El Capitan).