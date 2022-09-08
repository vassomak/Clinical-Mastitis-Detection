# Smart Farming Assisted by the Internet of Things for Animals’ Welfare

## Clinical mastitis detection based on on Udder Parameter using Internet of Things

Thesis report can be found [here](https://ikee.lib.auth.gr/record/338189/files/makrylaki.pdf). The part related to clinical mastitis is in chapter 5.

The data used for the project are derived from [here](https://data.mendeley.com/datasets/kbvcdw5b4m/1).

## Thesis Abstract

Nowadays, Internet of Things (IoT) is an integral part of people’s daily lives and modern’s economy. One of the areas that IoT has found great application, the last few years, is agriculture.
Via IoT many parameters can be controlled without constant human intervention, such as irrigation water, environmental conditions, and animals’ health. Especially over the next decade,
5G technology will contribute in supporting IoT sensor connectivity to the next level with wide
coverage, low energy consumption, low-cost devices, and high spectrum efficiency. However,
Cloud Computing seems incapable to satisfy the bandwidth, privacy and security requirements,
due to the increase of the connected IoT devices, which leads to excessive data quantity. For this
reason, academic and industrial research focuses on edge-based technologies, like Long Range
(LoRa), which migrates data computation to the network edge. LoRa belongs to the Low-Power
Wide-Area Netwrok (LPWAN) technologies and promises to provide high coverage with low energy consumption. The important disadvantage of LoRa seems to be the generic low bit-rate
(up to 5.5 Kbps). An appealing solution to this issue is the energy efficient version of Bluetooth,
known as Bluetooth Low Energy (BLE), as its data rate can be up to 1Mbps.
This thesis examines a hybrid wireless network of collared dairy cows equipped with sensors,
that are coupled to both LoRa and BLE radio platform in order to monitor animals’ location.
In particular, an optimization problem is formulated, that aims to maximize the average total
throughput of the system by optimizing the allocated packet transmission rate and the percentage
of nodes that use each LoRa Spreading Factor (SF). For the optimised values of these parameters,
the system is compared to the corresponding optimised system where only LoRa is used, as far
as throughput, success probability and energy consumption are concerned.
The second part of the thesis focuses exclusively on monitoring cattle’s well-being. Livestock
diseases can be disastrous, even if a single animal is infected. Thus, the need of automated
processes to detect whether an animal suffers from a disease is imperative. For this reason, we
aim to build Machine Learning models that are able to detect if a dairy cow suffers from clinical
mastitis. More specifically, a dataset of 6600 cattle measurements is used, which contains sensory
parameters and cows’ mastitis status. Supervised machine learning approaches were deployed
to determine the most effective parameters that could be utilized to predict the risk of clinical
mastitis in cattle. To achieve this purpose, 4 models were built using Support Vector Machines,
K-Nearest Neighbour, Decision Trees and Naive Bayes Classifier. Hyper parameter tuning and
K-fold cross validation were applied to enhance the models’ performance, while avoiding bias and
overfitting. The models’ ability to detect clinical mastitis were evaluated based on the Accuracy,
Precision, Recall and F-Score metrics.
