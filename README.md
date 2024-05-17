# NIDS
all you need to do is run [NIDS.ipynb](NIDS.ipynb) in google colab


Abstract
In the domain of network communication, network intrusion detection systems (NIDS) play a crucial role in maintaining security by identifying potential threats. NIDS relies on packet inspection, often using rule-based databases to scan for malicious patterns. However, the expanding scale of internet connections hampers the rate of packet inspection. To address this, some systems employ GPU accelerated pattern matching algorithms. Yet, this approach is susceptible to denial of service (DOS) attacks, inducing hashing collisions and slowing inspection. This research introduces a GPU-optimized variation of the Rabin-Karp algorithm, achieving scalability on GPUs while resisting DOS attacks. Our open-source solution (https://github.com/AnasAbbas1/NIDS) combines six polynomial hashing functions, eliminating the need for false-positive validation. This leads to a substantial improvement in inspection speed and accuracy. The proposed system ensures minimal packet misclassification rates, solidifying its role as a robust tool for real-time network security.

paper -> https://www.igi-global.com/gateway/article/full-text-pdf/341269&riu=true
