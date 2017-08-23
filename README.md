# GPU-Training-Logistic-Regression
We propose a new GPU implementation of logistic regression training method which provides both task and data parallelization. Logistic regression has been applied in many machine learning applications to build building predictive models. However, logistic training regularly requires a long time to adapt an accurate prediction model. Researchers have worked out to reduce training time using different technologies such as multi-threading, Multi-core CPUs and Message Passing Interface (MPI). In our study, the authors consider the high computation capabilities of GPU and easy development onto Open Computing Language (OpenCL) framework to execute logistic training process. GPU and OpenCL are the best choice with low cost and high performance for scaling up logistic regression model in handling large datasets. The proposed approach was implement in OpenCL C/C++ and tested by different size datasets on two GPU platforms. The experimental results showed a significant improvement in execution time with large datasets, which is reduced inversely by the available GPU computing units.

Feel free to contact me for further details and  If you are using this code, please cite our paper: 

Zahera, H. M., & El-Sisi, A. B. (2017). Accelerating Training Process in Logistic Regression Model using OpenCL Framework. International Journal of Grid and High Performance Computing (IJGHPC), 9(3), 34-45. doi:10.4018/IJGHPC.2017070103
