# Attention Shifting Distillation
This research study on **defense** against [Anti-Distillation-Backdoor-Attack](https://dl.acm.org/doi/abs/10.1145/3474085.3475254)(MM '21). Through our proposed Attention-Shifting-Distillation, we achieved a significant decrease in the attack success rate compared with [Backdoor Cleansing with Unlabeled Data](https://arxiv.org/abs/2211.12044)(CVPR '23) with a moderate decrease on the model accuracy.

## Table of Contents
* [Introduction](#introduction)
* [Attention Shifting Loss](#asl)
* [Results](#results)
* [Code Description](#code-description)
* [Installation](#installation)
* [Usage](#usage)


## Introduction<a name="introduction"></a>
**Knowledge distillation** (KD) is a well-established technique that enables the transfer of knowledge from a well-trained neural network (i.e., the teacher model) to another network (i.e., the student model), even when unlabeled data is used. Consequently, much research has been devoted to examining the potential risks of malicious behaviors being transferred during the KD process. One such malicious behavior is the **trigger backdoor attack**, in which a neural network produces incorrect outputs whenever the input contains a specific pattern (i.e., a trigger). **Anti-Distillation-Backdoor-Attack** (ADBA) is a study demonstrating that trigger backdoor attacks can be carried over to the student model through a particular training process. To counter this, we propose an improved knowledge distillation method called **Attention-Shifting-Distillation** (ASD), which aims to reduce the poison rate (i.e., attack success rate (ASR)) during KD from an ADBA model while retaining as much correct knowledge from the teacher model as possible.

## Attention Shifting Loss<a name="asl"></a>
TODO

## Results<a name="results"></a>
TODO

## Code Description<a name="code-description"></a>
TODO

## Installation<a name="installation"></a>
TODO

## Usage<a name="usage"></a>
TODO
