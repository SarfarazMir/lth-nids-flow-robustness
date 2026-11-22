
### RQ2 ddos19_d1→ddos19_d2, target label fraction = 0.01 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 1 | 0.9969 ± nan | 0.9968 ± nan | 0.0418 ± nan |
| dense_scratch | 0 | 0.000 | 1 | 0.9968 ± nan | 0.9966 ± nan | 0.6417 ± nan |
| rand_rewind | 7 | 0.790 | 1 | 0.9963 ± nan | 0.9962 ± nan | 0.6516 ± nan |
| rand_rewind | 10 | 0.893 | 1 | 0.9978 ± nan | 0.9977 ± nan | 0.4784 ± nan |
| rand_rewind | 13 | 0.945 | 1 | 0.9980 ± nan | 0.9979 ± nan | 0.7723 ± nan |
| rand_rewind | 16 | 0.972 | 1 | 0.8851 ± nan | 0.8881 ± nan | 0.4194 ± nan |
| snip_tgt | 7 | 0.790 | 1 | 0.9980 ± nan | 0.9979 ± nan | 0.6580 ± nan |
| snip_tgt | 10 | 0.893 | 1 | 0.9980 ± nan | 0.9980 ± nan | 0.7960 ± nan |
| snip_tgt | 13 | 0.945 | 1 | 0.9967 ± nan | 0.9966 ± nan | 0.6118 ± nan |
| snip_tgt | 16 | 0.972 | 1 | 0.9956 ± nan | 0.9954 ± nan | 0.8877 ± nan |
| tkt_ft | 7 | 0.790 | 1 | 0.9960 ± nan | 0.9958 ± nan | 0.0537 ± nan |
| tkt_ft | 10 | 0.893 | 1 | 0.9967 ± nan | 0.9966 ± nan | 0.0616 ± nan |
| tkt_ft | 13 | 0.945 | 1 | 0.9959 ± nan | 0.9957 ± nan | 0.0472 ± nan |
| tkt_ft | 16 | 0.972 | 1 | 0.9861 ± nan | 0.9854 ± nan | 0.1005 ± nan |
| tkt_reinit | 7 | 0.790 | 1 | 0.9975 ± nan | 0.9974 ± nan | 0.2685 ± nan |
| tkt_reinit | 10 | 0.893 | 1 | 0.9980 ± nan | 0.9979 ± nan | 0.2186 ± nan |
| tkt_reinit | 13 | 0.945 | 1 | 0.9983 ± nan | 0.9983 ± nan | 0.2775 ± nan |
| tkt_reinit | 16 | 0.972 | 1 | 0.9747 ± nan | 0.9752 ± nan | 0.9415 ± nan |
| tkt_rewind | 7 | 0.790 | 1 | 0.9968 ± nan | 0.9966 ± nan | 0.0480 ± nan |
| tkt_rewind | 10 | 0.893 | 1 | 0.9966 ± nan | 0.9965 ± nan | 0.0820 ± nan |
| tkt_rewind | 13 | 0.945 | 1 | 0.9965 ± nan | 0.9963 ± nan | 0.0578 ± nan |
| tkt_rewind | 16 | 0.972 | 1 | 0.9930 ± nan | 0.9927 ± nan | 0.5863 ± nan |

### RQ2 ddos19_d1→ddos19_d2, target label fraction = 0.1 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 1 | 0.9983 ± nan | 0.9983 ± nan | 0.0727 ± nan |
| dense_scratch | 0 | 0.000 | 1 | 0.9990 ± nan | 0.9990 ± nan | 0.7894 ± nan |
| rand_rewind | 7 | 0.790 | 1 | 0.9987 ± nan | 0.9987 ± nan | 0.8592 ± nan |
| rand_rewind | 10 | 0.893 | 1 | 0.9986 ± nan | 0.9986 ± nan | 0.8132 ± nan |
| rand_rewind | 13 | 0.945 | 1 | 0.9985 ± nan | 0.9984 ± nan | 0.5827 ± nan |
| rand_rewind | 16 | 0.972 | 1 | 0.9151 ± nan | 0.9179 ± nan | 0.9467 ± nan |
| snip_tgt | 7 | 0.790 | 1 | 0.9987 ± nan | 0.9987 ± nan | 0.6482 ± nan |
| snip_tgt | 10 | 0.893 | 1 | 0.9986 ± nan | 0.9986 ± nan | 0.8447 ± nan |
| snip_tgt | 13 | 0.945 | 1 | 0.9984 ± nan | 0.9984 ± nan | 0.8528 ± nan |
| snip_tgt | 16 | 0.972 | 1 | 0.9980 ± nan | 0.9980 ± nan | 0.9394 ± nan |
| tkt_ft | 7 | 0.790 | 1 | 0.9979 ± nan | 0.9978 ± nan | 0.0642 ± nan |
| tkt_ft | 10 | 0.893 | 1 | 0.9975 ± nan | 0.9974 ± nan | 0.0792 ± nan |
| tkt_ft | 13 | 0.945 | 1 | 0.9974 ± nan | 0.9973 ± nan | 0.0619 ± nan |
| tkt_ft | 16 | 0.972 | 1 | 0.9880 ± nan | 0.9874 ± nan | 0.1078 ± nan |
| tkt_reinit | 7 | 0.790 | 1 | 0.9987 ± nan | 0.9987 ± nan | 0.6029 ± nan |
| tkt_reinit | 10 | 0.893 | 1 | 0.9989 ± nan | 0.9988 ± nan | 0.4543 ± nan |
| tkt_reinit | 13 | 0.945 | 1 | 0.9983 ± nan | 0.9983 ± nan | 0.4987 ± nan |
| tkt_reinit | 16 | 0.972 | 1 | 0.9591 ± nan | 0.9603 ± nan | 0.4598 ± nan |
| tkt_rewind | 7 | 0.790 | 1 | 0.9983 ± nan | 0.9983 ± nan | 0.3220 ± nan |
| tkt_rewind | 10 | 0.893 | 1 | 0.9982 ± nan | 0.9981 ± nan | 0.5436 ± nan |
| tkt_rewind | 13 | 0.945 | 1 | 0.9982 ± nan | 0.9981 ± nan | 0.3396 ± nan |
| tkt_rewind | 16 | 0.972 | 1 | 0.9942 ± nan | 0.9939 ± nan | 0.9116 ± nan |

### RQ2 ddos19_d1→ddos19_d2, target label fraction = 1.0 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.2559 ± nan |
| dense_scratch | 0 | 0.000 | 1 | 0.9999 ± nan | 0.9999 ± nan | 0.8758 ± nan |
| rand_rewind | 7 | 0.790 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.8969 ± nan |
| rand_rewind | 10 | 0.893 | 1 | 0.9997 ± nan | 0.9997 ± nan | 0.8927 ± nan |
| rand_rewind | 13 | 0.945 | 1 | 0.9993 ± nan | 0.9993 ± nan | 0.7775 ± nan |
| rand_rewind | 16 | 0.972 | 1 | 0.9933 ± nan | 0.9932 ± nan | 0.9072 ± nan |
| snip_tgt | 7 | 0.790 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.7826 ± nan |
| snip_tgt | 10 | 0.893 | 1 | 0.9993 ± nan | 0.9993 ± nan | 0.7874 ± nan |
| snip_tgt | 13 | 0.945 | 1 | 0.9993 ± nan | 0.9993 ± nan | 0.6285 ± nan |
| snip_tgt | 16 | 0.972 | 1 | 0.9991 ± nan | 0.9991 ± nan | 0.8859 ± nan |
| tkt_ft | 7 | 0.790 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.0788 ± nan |
| tkt_ft | 10 | 0.893 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.1265 ± nan |
| tkt_ft | 13 | 0.945 | 1 | 0.9989 ± nan | 0.9988 ± nan | 0.0657 ± nan |
| tkt_ft | 16 | 0.972 | 1 | 0.9900 ± nan | 0.9895 ± nan | 0.1350 ± nan |
| tkt_reinit | 7 | 0.790 | 1 | 0.9999 ± nan | 0.9999 ± nan | 0.8191 ± nan |
| tkt_reinit | 10 | 0.893 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.7783 ± nan |
| tkt_reinit | 13 | 0.945 | 1 | 0.9992 ± nan | 0.9991 ± nan | 0.5747 ± nan |
| tkt_reinit | 16 | 0.972 | 1 | 0.9969 ± nan | 0.9967 ± nan | 0.9564 ± nan |
| tkt_rewind | 7 | 0.790 | 1 | 1.0000 ± nan | 1.0000 ± nan | 0.4495 ± nan |
| tkt_rewind | 10 | 0.893 | 1 | 0.9998 ± nan | 0.9998 ± nan | 0.6536 ± nan |
| tkt_rewind | 13 | 0.945 | 1 | 0.9997 ± nan | 0.9997 ± nan | 0.5110 ± nan |
| tkt_rewind | 16 | 0.972 | 1 | 0.9957 ± nan | 0.9955 ± nan | 0.9037 ± nan |

### RQ2 ddos19_d1→ids17, target label fraction = 0.01 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 5 | 0.9678 ± 0.0033 | 0.9596 ± 0.0032 | 0.4600 ± 0.0222 |
| dense_scratch | 0 | 0.000 | 5 | 0.9763 ± 0.0031 | 0.9758 ± 0.0031 | 0.5882 ± 0.0372 |
| rand_rewind | 7 | 0.790 | 1 | 0.9725 ± nan | 0.9741 ± nan | 0.6362 ± nan |
| rand_rewind | 10 | 0.893 | 5 | 0.9765 ± 0.0026 | 0.9765 ± 0.0026 | 0.5919 ± 0.0419 |
| rand_rewind | 13 | 0.945 | 5 | 0.9737 ± 0.0019 | 0.9727 ± 0.0016 | 0.6767 ± 0.0768 |
| rand_rewind | 16 | 0.972 | 5 | 0.6074 ± 0.1492 | 0.4961 ± 0.2211 | 0.0003 ± 0.0004 |
| snip_tgt | 7 | 0.790 | 1 | 0.9725 ± nan | 0.9745 ± nan | 0.6258 ± nan |
| snip_tgt | 10 | 0.893 | 5 | 0.9758 ± 0.0027 | 0.9752 ± 0.0028 | 0.6191 ± 0.0524 |
| snip_tgt | 13 | 0.945 | 5 | 0.9761 ± 0.0026 | 0.9752 ± 0.0024 | 0.6038 ± 0.0452 |
| snip_tgt | 16 | 0.972 | 5 | 0.9738 ± 0.0021 | 0.9728 ± 0.0029 | 0.5927 ± 0.0663 |
| tkt_ft | 7 | 0.790 | 1 | 0.9403 ± nan | 0.9167 ± nan | 0.4145 ± nan |
| tkt_ft | 10 | 0.893 | 5 | 0.9386 ± 0.0059 | 0.9074 ± 0.0105 | 0.3415 ± 0.0339 |
| tkt_ft | 13 | 0.945 | 5 | 0.9127 ± 0.0030 | 0.8602 ± 0.0037 | 0.2893 ± 0.0910 |
| tkt_ft | 16 | 0.972 | 5 | 0.4323 ± 0.0279 | 0.3504 ± 0.0361 | 0.1800 ± 0.0610 |
| tkt_reinit | 7 | 0.790 | 1 | 0.9715 ± nan | 0.9723 ± nan | 0.5857 ± nan |
| tkt_reinit | 10 | 0.893 | 5 | 0.9764 ± 0.0028 | 0.9766 ± 0.0024 | 0.5932 ± 0.0557 |
| tkt_reinit | 13 | 0.945 | 5 | 0.9746 ± 0.0018 | 0.9742 ± 0.0019 | 0.6536 ± 0.0717 |
| tkt_reinit | 16 | 0.972 | 5 | 0.9249 ± 0.0170 | 0.8893 ± 0.0268 | 0.6177 ± 0.1389 |
| tkt_rewind | 7 | 0.790 | 1 | 0.9716 ± nan | 0.9716 ± nan | 0.6330 ± nan |
| tkt_rewind | 10 | 0.893 | 5 | 0.9747 ± 0.0021 | 0.9736 ± 0.0022 | 0.5665 ± 0.0516 |
| tkt_rewind | 13 | 0.945 | 5 | 0.9727 ± 0.0018 | 0.9713 ± 0.0013 | 0.6461 ± 0.0417 |
| tkt_rewind | 16 | 0.972 | 5 | 0.8964 ± 0.0150 | 0.8617 ± 0.0285 | 0.3946 ± 0.2530 |

### RQ2 ddos19_d1→ids17, target label fraction = 0.1 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 1 | 0.9724 ± nan | 0.9626 ± nan | 0.4739 ± nan |
| dense_scratch | 0 | 0.000 | 1 | 0.9886 ± nan | 0.9877 ± nan | 0.5092 ± nan |
| rand_rewind | 7 | 0.790 | 1 | 0.9882 ± nan | 0.9868 ± nan | 0.5744 ± nan |
| rand_rewind | 10 | 0.893 | 1 | 0.9880 ± nan | 0.9866 ± nan | 0.5140 ± nan |
| rand_rewind | 13 | 0.945 | 1 | 0.9849 ± nan | 0.9826 ± nan | 0.5771 ± nan |
| rand_rewind | 16 | 0.972 | 1 | 0.8842 ± nan | 0.8645 ± nan | 0.8410 ± nan |
| snip_tgt | 7 | 0.790 | 1 | 0.9881 ± nan | 0.9873 ± nan | 0.5813 ± nan |
| snip_tgt | 10 | 0.893 | 1 | 0.9878 ± nan | 0.9868 ± nan | 0.5203 ± nan |
| snip_tgt | 13 | 0.945 | 1 | 0.9877 ± nan | 0.9862 ± nan | 0.5359 ± nan |
| snip_tgt | 16 | 0.972 | 1 | 0.9858 ± nan | 0.9834 ± nan | 0.5070 ± nan |
| tkt_ft | 7 | 0.790 | 1 | 0.9460 ± nan | 0.9174 ± nan | 0.3821 ± nan |
| tkt_ft | 10 | 0.893 | 1 | 0.9304 ± nan | 0.8925 ± nan | 0.3023 ± nan |
| tkt_ft | 13 | 0.945 | 1 | 0.9131 ± nan | 0.8614 ± nan | 0.3844 ± nan |
| tkt_ft | 16 | 0.972 | 1 | 0.4378 ± nan | 0.3483 ± nan | 0.1663 ± nan |
| tkt_reinit | 7 | 0.790 | 1 | 0.9883 ± nan | 0.9870 ± nan | 0.5345 ± nan |
| tkt_reinit | 10 | 0.893 | 1 | 0.9883 ± nan | 0.9869 ± nan | 0.5326 ± nan |
| tkt_reinit | 13 | 0.945 | 1 | 0.9860 ± nan | 0.9840 ± nan | 0.6251 ± nan |
| tkt_reinit | 16 | 0.972 | 1 | 0.9526 ± nan | 0.9307 ± nan | 0.6477 ± nan |
| tkt_rewind | 7 | 0.790 | 1 | 0.9849 ± nan | 0.9837 ± nan | 0.5211 ± nan |
| tkt_rewind | 10 | 0.893 | 1 | 0.9860 ± nan | 0.9834 ± nan | 0.5471 ± nan |
| tkt_rewind | 13 | 0.945 | 1 | 0.9820 ± nan | 0.9795 ± nan | 0.5749 ± nan |
| tkt_rewind | 16 | 0.972 | 1 | 0.9168 ± nan | 0.8775 ± nan | 0.1896 ± nan |

### RQ2 ddos19_d1→ids17, target label fraction = 1.0 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 5 | 0.9899 ± 0.0004 | 0.9885 ± 0.0002 | 0.4860 ± 0.0452 |
| dense_scratch | 0 | 0.000 | 5 | 0.9918 ± 0.0004 | 0.9913 ± 0.0002 | 0.4461 ± 0.0386 |
| rand_rewind | 7 | 0.790 | 1 | 0.9912 ± nan | 0.9909 ± nan | 0.4656 ± nan |
| rand_rewind | 10 | 0.893 | 5 | 0.9912 ± 0.0004 | 0.9908 ± 0.0003 | 0.4627 ± 0.0455 |
| rand_rewind | 13 | 0.945 | 5 | 0.9907 ± 0.0002 | 0.9901 ± 0.0002 | 0.4894 ± 0.0382 |
| rand_rewind | 16 | 0.972 | 5 | 0.6480 ± 0.2052 | 0.5682 ± 0.3125 | 0.5654 ± 0.4911 |
| snip_tgt | 7 | 0.790 | 1 | 0.9912 ± nan | 0.9907 ± nan | 0.4810 ± nan |
| snip_tgt | 10 | 0.893 | 5 | 0.9913 ± 0.0003 | 0.9909 ± 0.0001 | 0.4935 ± 0.0749 |
| snip_tgt | 13 | 0.945 | 5 | 0.9913 ± 0.0003 | 0.9907 ± 0.0002 | 0.4615 ± 0.0195 |
| snip_tgt | 16 | 0.972 | 5 | 0.9907 ± 0.0004 | 0.9901 ± 0.0006 | 0.4638 ± 0.0635 |
| tkt_ft | 7 | 0.790 | 1 | 0.9852 ± nan | 0.9833 ± nan | 0.4692 ± nan |
| tkt_ft | 10 | 0.893 | 5 | 0.9848 ± 0.0004 | 0.9828 ± 0.0004 | 0.4823 ± 0.0745 |
| tkt_ft | 13 | 0.945 | 5 | 0.9808 ± 0.0009 | 0.9771 ± 0.0013 | 0.5221 ± 0.0825 |
| tkt_ft | 16 | 0.972 | 5 | 0.9016 ± 0.0343 | 0.8709 ± 0.0507 | 0.3401 ± 0.1944 |
| tkt_reinit | 7 | 0.790 | 1 | 0.9918 ± nan | 0.9912 ± nan | 0.5273 ± nan |
| tkt_reinit | 10 | 0.893 | 5 | 0.9911 ± 0.0001 | 0.9907 ± 0.0002 | 0.4632 ± 0.0109 |
| tkt_reinit | 13 | 0.945 | 5 | 0.9906 ± 0.0001 | 0.9900 ± 0.0002 | 0.4647 ± 0.0305 |
| tkt_reinit | 16 | 0.972 | 5 | 0.9721 ± 0.0031 | 0.9628 ± 0.0039 | 0.5346 ± 0.1222 |
| tkt_rewind | 7 | 0.790 | 1 | 0.9916 ± nan | 0.9908 ± nan | 0.5474 ± nan |
| tkt_rewind | 10 | 0.893 | 5 | 0.9910 ± 0.0002 | 0.9905 ± 0.0002 | 0.4803 ± 0.0437 |
| tkt_rewind | 13 | 0.945 | 5 | 0.9904 ± 0.0003 | 0.9897 ± 0.0003 | 0.5686 ± 0.0318 |
| tkt_rewind | 16 | 0.972 | 5 | 0.9667 ± 0.0034 | 0.9568 ± 0.0054 | 0.5024 ± 0.1179 |

### RQ2 ids17→ddos19_d1, target label fraction = 0.01 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 5 | 0.9949 ± 0.0009 | 0.9942 ± 0.0009 | 0.3818 ± 0.2186 |
| dense_scratch | 0 | 0.000 | 5 | 0.9950 ± 0.0004 | 0.9945 ± 0.0008 | 0.0659 ± 0.0920 |
| rand_rewind | 7 | 0.790 | 1 | 0.9946 ± nan | 0.9937 ± nan | 0.0852 ± nan |
| rand_rewind | 10 | 0.893 | 5 | 0.9953 ± 0.0009 | 0.9949 ± 0.0012 | 0.0362 ± 0.0465 |
| rand_rewind | 13 | 0.945 | 5 | 0.9952 ± 0.0010 | 0.9947 ± 0.0011 | 0.0135 ± 0.0046 |
| rand_rewind | 16 | 0.972 | 5 | 0.5850 ± 0.1900 | 0.5119 ± 0.2369 | 0.1439 ± 0.3218 |
| snip_tgt | 7 | 0.790 | 1 | 0.9951 ± nan | 0.9942 ± nan | 0.1130 ± nan |
| snip_tgt | 10 | 0.893 | 5 | 0.9954 ± 0.0007 | 0.9948 ± 0.0009 | 0.0426 ± 0.0428 |
| snip_tgt | 13 | 0.945 | 5 | 0.9956 ± 0.0006 | 0.9952 ± 0.0008 | 0.1975 ± 0.3467 |
| snip_tgt | 16 | 0.972 | 5 | 0.9952 ± 0.0007 | 0.9945 ± 0.0010 | 0.0186 ± 0.0191 |
| tkt_ft | 7 | 0.790 | 1 | 0.9924 ± nan | 0.9902 ± nan | 0.7617 ± nan |
| tkt_ft | 10 | 0.893 | 5 | 0.9907 ± 0.0019 | 0.9868 ± 0.0029 | 0.6887 ± 0.2490 |
| tkt_ft | 13 | 0.945 | 5 | 0.9738 ± 0.0044 | 0.9621 ± 0.0061 | 0.8787 ± 0.0551 |
| tkt_ft | 16 | 0.972 | 5 | 0.6791 ± 0.0408 | 0.5867 ± 0.0532 | 0.6932 ± 0.2664 |
| tkt_reinit | 7 | 0.790 | 1 | 0.9948 ± nan | 0.9939 ± nan | 0.0270 ± nan |
| tkt_reinit | 10 | 0.893 | 5 | 0.9950 ± 0.0013 | 0.9945 ± 0.0013 | 0.0183 ± 0.0130 |
| tkt_reinit | 13 | 0.945 | 5 | 0.9950 ± 0.0009 | 0.9944 ± 0.0010 | 0.0458 ± 0.0271 |
| tkt_reinit | 16 | 0.972 | 5 | 0.9570 ± 0.0152 | 0.9391 ± 0.0243 | 0.1248 ± 0.0986 |
| tkt_rewind | 7 | 0.790 | 1 | 0.9950 ± nan | 0.9943 ± nan | 0.1478 ± nan |
| tkt_rewind | 10 | 0.893 | 5 | 0.9949 ± 0.0010 | 0.9943 ± 0.0012 | 0.0600 ± 0.1038 |
| tkt_rewind | 13 | 0.945 | 5 | 0.9950 ± 0.0008 | 0.9946 ± 0.0010 | 0.0425 ± 0.0368 |
| tkt_rewind | 16 | 0.972 | 5 | 0.9716 ± 0.0033 | 0.9632 ± 0.0087 | 0.2993 ± 0.2591 |

### RQ2 ids17→ddos19_d1, target label fraction = 0.1 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 1 | 0.9960 ± nan | 0.9957 ± nan | 0.4523 ± nan |
| dense_scratch | 0 | 0.000 | 1 | 0.9970 ± nan | 0.9966 ± nan | 0.5541 ± nan |
| rand_rewind | 7 | 0.790 | 1 | 0.9969 ± nan | 0.9965 ± nan | 0.3563 ± nan |
| rand_rewind | 10 | 0.893 | 1 | 0.9970 ± nan | 0.9966 ± nan | 0.6569 ± nan |
| rand_rewind | 13 | 0.945 | 1 | 0.9966 ± nan | 0.9963 ± nan | 0.0149 ± nan |
| rand_rewind | 16 | 0.972 | 1 | 0.5000 ± nan | 0.2405 ± nan | nan ± nan |
| snip_tgt | 7 | 0.790 | 1 | 0.9970 ± nan | 0.9967 ± nan | 0.2145 ± nan |
| snip_tgt | 10 | 0.893 | 1 | 0.9966 ± nan | 0.9963 ± nan | 0.2278 ± nan |
| snip_tgt | 13 | 0.945 | 1 | 0.9970 ± nan | 0.9966 ± nan | 0.4800 ± nan |
| snip_tgt | 16 | 0.972 | 1 | 0.9962 ± nan | 0.9960 ± nan | 0.1297 ± nan |
| tkt_ft | 7 | 0.790 | 1 | 0.9947 ± nan | 0.9938 ± nan | 0.7501 ± nan |
| tkt_ft | 10 | 0.893 | 1 | 0.9946 ± nan | 0.9933 ± nan | 0.8603 ± nan |
| tkt_ft | 13 | 0.945 | 1 | 0.9826 ± nan | 0.9750 ± nan | 0.8044 ± nan |
| tkt_ft | 16 | 0.972 | 1 | 0.6439 ± nan | 0.5369 ± nan | 0.8699 ± nan |
| tkt_reinit | 7 | 0.790 | 1 | 0.9970 ± nan | 0.9966 ± nan | 0.0409 ± nan |
| tkt_reinit | 10 | 0.893 | 1 | 0.9970 ± nan | 0.9966 ± nan | 0.1093 ± nan |
| tkt_reinit | 13 | 0.945 | 1 | 0.9966 ± nan | 0.9962 ± nan | 0.0607 ± nan |
| tkt_reinit | 16 | 0.972 | 1 | 0.9723 ± nan | 0.9669 ± nan | 0.0076 ± nan |
| tkt_rewind | 7 | 0.790 | 1 | 0.9968 ± nan | 0.9964 ± nan | 0.1633 ± nan |
| tkt_rewind | 10 | 0.893 | 1 | 0.9967 ± nan | 0.9964 ± nan | 0.4408 ± nan |
| tkt_rewind | 13 | 0.945 | 1 | 0.9964 ± nan | 0.9961 ± nan | 0.1457 ± nan |
| tkt_rewind | 16 | 0.972 | 1 | 0.9729 ± nan | 0.9659 ± nan | 0.3073 ± nan |

### RQ2 ids17→ddos19_d1, target label fraction = 1.0 (mean ± sd over seeds)

| cond | round | sparsity | n | tgt_test/bacc | tgt_test/macro_f1 | flow/ESR_AUC |
|---|---|---|---|---|---|---|
| dense_ft | 0 | 0.000 | 5 | 0.9972 ± 0.0001 | 0.9967 ± 0.0002 | 0.3442 ± 0.1386 |
| dense_scratch | 0 | 0.000 | 5 | 0.9982 ± 0.0001 | 0.9980 ± 0.0001 | 0.2022 ± 0.0888 |
| rand_rewind | 7 | 0.790 | 1 | 0.9981 ± nan | 0.9978 ± nan | 0.2555 ± nan |
| rand_rewind | 10 | 0.893 | 5 | 0.9980 ± 0.0000 | 0.9977 ± 0.0001 | 0.3563 ± 0.1994 |
| rand_rewind | 13 | 0.945 | 5 | 0.9976 ± 0.0002 | 0.9973 ± 0.0001 | 0.2541 ± 0.0854 |
| rand_rewind | 16 | 0.972 | 5 | 0.5947 ± 0.2118 | 0.4857 ± 0.2802 | 0.2427 ± 0.4855 |
| snip_tgt | 7 | 0.790 | 1 | 0.9980 ± nan | 0.9977 ± nan | 0.7474 ± nan |
| snip_tgt | 10 | 0.893 | 5 | 0.9981 ± 0.0001 | 0.9977 ± 0.0002 | 0.3436 ± 0.1190 |
| snip_tgt | 13 | 0.945 | 5 | 0.9978 ± 0.0003 | 0.9975 ± 0.0003 | 0.4223 ± 0.3029 |
| snip_tgt | 16 | 0.972 | 5 | 0.9973 ± 0.0001 | 0.9970 ± 0.0002 | 0.5392 ± 0.2609 |
| tkt_ft | 7 | 0.790 | 1 | 0.9966 ± nan | 0.9961 ± nan | 0.6352 ± nan |
| tkt_ft | 10 | 0.893 | 5 | 0.9961 ± 0.0001 | 0.9956 ± 0.0002 | 0.5731 ± 0.2424 |
| tkt_ft | 13 | 0.945 | 5 | 0.9958 ± 0.0005 | 0.9953 ± 0.0006 | 0.5745 ± 0.1694 |
| tkt_ft | 16 | 0.972 | 5 | 0.9231 ± 0.0232 | 0.9009 ± 0.0313 | 0.6586 ± 0.1902 |
| tkt_reinit | 7 | 0.790 | 1 | 0.9982 ± nan | 0.9980 ± nan | 0.0362 ± nan |
| tkt_reinit | 10 | 0.893 | 5 | 0.9980 ± 0.0001 | 0.9977 ± 0.0001 | 0.2576 ± 0.1276 |
| tkt_reinit | 13 | 0.945 | 5 | 0.9976 ± 0.0002 | 0.9973 ± 0.0002 | 0.2535 ± 0.0786 |
| tkt_reinit | 16 | 0.972 | 5 | 0.9836 ± 0.0064 | 0.9805 ± 0.0059 | 0.5542 ± 0.4422 |
| tkt_rewind | 7 | 0.790 | 1 | 0.9980 ± nan | 0.9976 ± nan | 0.2328 ± nan |
| tkt_rewind | 10 | 0.893 | 5 | 0.9979 ± 0.0002 | 0.9975 ± 0.0002 | 0.2388 ± 0.2121 |
| tkt_rewind | 13 | 0.945 | 5 | 0.9972 ± 0.0001 | 0.9970 ± 0.0001 | 0.2937 ± 0.1093 |
| tkt_rewind | 16 | 0.972 | 5 | 0.9837 ± 0.0048 | 0.9801 ± 0.0065 | 0.2620 ± 0.1957 |

### Paired Wilcoxon vs tkt_rewind (pairs = seed × sparsity; Holm-corrected)

| pair                |   frac | metric            | cond        | ref        |   n_pairs |   median_diff |   mean_diff |   wins |         p |    p_holm |
|:--------------------|-------:|:------------------|:------------|:-----------|----------:|--------------:|------------:|-------:|----------:|----------:|
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/bacc     | rand_rewind | tkt_rewind |         4 |     0.0003654 |  -0.02642   |      2 | 1         | 1         |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/bacc     | snip_tgt    | tkt_rewind |         4 |     0.001281  |   0.001343  |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/bacc     | tkt_ft      | tkt_rewind |         4 |    -0.0006896 |  -0.00204   |      1 | 0.25      | 0.75      |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/bacc     | tkt_reinit  | tkt_rewind |         4 |     0.001043  |  -0.003585  |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/macro_f1 | rand_rewind | tkt_rewind |         4 |     0.0003903 |  -0.02558   |      2 | 1         | 1         |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |         4 |     0.001366  |   0.001405  |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |         4 |    -0.0007413 |  -0.002183  |      1 | 0.25      | 0.75      |
| ddos19_d1→ddos19_d2 |   0.01 | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |         4 |     0.001093  |  -0.003347  |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.01 | flow/ESR_AUC      | rand_rewind | tkt_rewind |         4 |     0.5       |   0.3869    |      3 | 0.25      | 0.5       |
| ddos19_d1→ddos19_d2 |   0.01 | flow/ESR_AUC      | snip_tgt    | tkt_rewind |         4 |     0.582     |   0.5449    |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.01 | flow/ESR_AUC      | tkt_ft      | tkt_rewind |         4 |    -0.01549   |  -0.1278    |      1 | 0.25      | 0.5       |
| ddos19_d1→ddos19_d2 |   0.01 | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |         4 |     0.2201    |   0.233     |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/bacc     | rand_rewind | tkt_rewind |         4 |     0.000341  |  -0.01948   |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/bacc     | snip_tgt    | tkt_rewind |         4 |     0.0004065 |   0.001228  |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/bacc     | tkt_ft      | tkt_rewind |         4 |    -0.0007349 |  -0.00203   |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/bacc     | tkt_reinit  | tkt_rewind |         4 |     0.0002671 |  -0.008478  |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/macro_f1 | rand_rewind | tkt_rewind |         4 |     0.0003514 |  -0.01871   |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |         4 |     0.0004294 |   0.001288  |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |         4 |    -0.0007807 |  -0.002145  |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |         4 |     0.0002733 |  -0.00808   |      3 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   0.1  | flow/ESR_AUC      | rand_rewind | tkt_rewind |         4 |     0.2563    |   0.2712    |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | flow/ESR_AUC      | snip_tgt    | tkt_rewind |         4 |     0.3136    |   0.2921    |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | flow/ESR_AUC      | tkt_ft      | tkt_rewind |         4 |    -0.371     |  -0.4509    |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   0.1  | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |         4 |     0.03488   |  -0.02528   |      2 | 1         | 1         |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/bacc     | rand_rewind | tkt_rewind |         4 |    -0.0002998 |  -0.0007835 |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/bacc     | snip_tgt    | tkt_rewind |         4 |    -0.0002629 |   0.0005871 |      1 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/bacc     | tkt_ft      | tkt_rewind |         4 |    -0.0005216 |  -0.001684  |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/bacc     | tkt_reinit  | tkt_rewind |         4 |    -7.81e-05  |   0.0001293 |      1 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/macro_f1 | rand_rewind | tkt_rewind |         4 |    -0.0003124 |  -0.0007616 |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |         4 |    -0.0002733 |   0.0006244 |      1 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |         4 |    -0.0005467 |  -0.001775  |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |         4 |    -7.81e-05  |   0.0001365 |      1 | 0.875     | 1         |
| ddos19_d1→ddos19_d2 |   1    | flow/ESR_AUC      | rand_rewind | tkt_rewind |         4 |     0.2527    |   0.2391    |      4 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | flow/ESR_AUC      | snip_tgt    | tkt_rewind |         4 |     0.1256    |   0.1416    |      3 | 0.25      | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | flow/ESR_AUC      | tkt_ft      | tkt_rewind |         4 |    -0.4862    |  -0.528     |      0 | 0.125     | 0.5       |
| ddos19_d1→ddos19_d2 |   1    | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |         4 |     0.09414   |   0.1526    |      4 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.01 | tgt_test/bacc     | rand_rewind | tkt_rewind |        16 |     0.0005941 |  -0.08937   |     10 | 0.8603    | 0.8603    |
| ddos19_d1→ids17     |   0.01 | tgt_test/bacc     | snip_tgt    | tkt_rewind |        16 |     0.003038  |   0.02564   |     15 | 0.0001526 | 0.0004578 |
| ddos19_d1→ids17     |   0.01 | tgt_test/bacc     | tkt_ft      | tkt_rewind |        16 |    -0.05948   |  -0.177     |      0 | 3.052e-05 | 0.0001221 |
| ddos19_d1→ids17     |   0.01 | tgt_test/bacc     | tkt_reinit  | tkt_rewind |        16 |     0.002606  |   0.01      |     14 | 0.001007  | 0.002014  |
| ddos19_d1→ids17     |   0.01 | tgt_test/macro_f1 | rand_rewind | tkt_rewind |        16 |     0.001593  |  -0.1127    |     10 | 0.8603    | 0.8603    |
| ddos19_d1→ids17     |   0.01 | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |        16 |     0.003375  |   0.0366    |     16 | 3.052e-05 | 0.0001221 |
| ddos19_d1→ids17     |   0.01 | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |        16 |    -0.1117    |  -0.2186    |      0 | 3.052e-05 | 0.0001221 |
| ddos19_d1→ids17     |   0.01 | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |        16 |     0.003094  |   0.01051   |     15 | 0.002136  | 0.004272  |
| ddos19_d1→ids17     |   0.01 | flow/ESR_AUC      | rand_rewind | tkt_rewind |        14 |     0.006775  |  -0.06191   |      8 | 0.8552    | 0.8552    |
| ddos19_d1→ids17     |   0.01 | flow/ESR_AUC      | snip_tgt    | tkt_rewind |        16 |     0.0262    |   0.0647    |      9 | 0.3484    | 0.6968    |
| ddos19_d1→ids17     |   0.01 | flow/ESR_AUC      | tkt_ft      | tkt_rewind |        16 |    -0.2546    |  -0.2625    |      1 | 9.155e-05 | 0.0003662 |
| ddos19_d1→ids17     |   0.01 | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |        16 |     0.01233   |   0.07745   |     10 | 0.1439    | 0.4316    |
| ddos19_d1→ids17     |   0.1  | tgt_test/bacc     | rand_rewind | tkt_rewind |         4 |     0.002478  |  -0.006072  |      3 | 0.875     | 0.875     |
| ddos19_d1→ids17     |   0.1  | tgt_test/bacc     | snip_tgt    | tkt_rewind |         4 |     0.004438  |   0.01994   |      4 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | tgt_test/bacc     | tkt_ft      | tkt_rewind |         4 |    -0.06222   |  -0.1606    |      0 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | tgt_test/bacc     | tkt_reinit  | tkt_rewind |         4 |     0.003692  |   0.0114    |      4 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | tgt_test/macro_f1 | rand_rewind | tkt_rewind |         4 |     0.003126  |  -0.0008873 |      3 | 0.875     | 0.875     |
| ddos19_d1→ids17     |   0.1  | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |         4 |     0.005158  |   0.02989   |      4 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |         4 |    -0.1045    |  -0.2011    |      0 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |         4 |     0.003993  |   0.01612   |      4 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | flow/ESR_AUC      | rand_rewind | tkt_rewind |         4 |     0.02766   |   0.1684    |      3 | 0.375     | 1         |
| ddos19_d1→ids17     |   0.1  | flow/ESR_AUC      | snip_tgt    | tkt_rewind |         4 |     0.01666   |   0.07792   |      2 | 0.625     | 1         |
| ddos19_d1→ids17     |   0.1  | flow/ESR_AUC      | tkt_ft      | tkt_rewind |         4 |    -0.1648    |  -0.1494    |      0 | 0.125     | 0.5       |
| ddos19_d1→ids17     |   0.1  | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |         4 |     0.03178   |   0.1268    |      3 | 0.375     | 1         |
| ddos19_d1→ids17     |   1    | tgt_test/bacc     | rand_rewind | tkt_rewind |        16 |    -0.0001945 |  -0.09948   |      7 | 0.2312    | 0.2312    |
| ddos19_d1→ids17     |   1    | tgt_test/bacc     | snip_tgt    | tkt_rewind |        16 |     0.0007311 |   0.007837  |     14 | 0.001007  | 0.003021  |
| ddos19_d1→ids17     |   1    | tgt_test/bacc     | tkt_ft      | tkt_rewind |        16 |    -0.009609  |  -0.02569   |      0 | 3.052e-05 | 0.0001221 |
| ddos19_d1→ids17     |   1    | tgt_test/bacc     | tkt_reinit  | tkt_rewind |        16 |     0.0002617 |   0.001759  |     13 | 0.007629  | 0.01526   |
| ddos19_d1→ids17     |   1    | tgt_test/macro_f1 | rand_rewind | tkt_rewind |        16 |     0.0002048 |  -0.1212    |     10 | 0.8209    | 0.8209    |
| ddos19_d1→ids17     |   1    | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |        16 |     0.001002  |   0.01085   |     15 | 9.155e-05 | 0.0002747 |
| ddos19_d1→ids17     |   1    | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |        16 |    -0.01228   |  -0.03362   |      0 | 3.052e-05 | 0.0001221 |
| ddos19_d1→ids17     |   1    | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |        16 |     0.0005169 |   0.002079  |     14 | 0.0002136 | 0.0004272 |
| ddos19_d1→ids17     |   1    | flow/ESR_AUC      | rand_rewind | tkt_rewind |        14 |    -0.06082   |  -0.0159    |      4 | 0.2166    | 0.4229    |
| ddos19_d1→ids17     |   1    | flow/ESR_AUC      | snip_tgt    | tkt_rewind |        16 |    -0.0573    |  -0.04555   |      5 | 0.05066   | 0.2026    |
| ddos19_d1→ids17     |   1    | flow/ESR_AUC      | tkt_ft      | tkt_rewind |        16 |    -0.06382   |  -0.06952   |      6 | 0.1297    | 0.3892    |
| ddos19_d1→ids17     |   1    | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |        16 |    -0.03363   |  -0.02898   |      5 | 0.2114    | 0.4229    |
| ids17→ddos19_d1     |   0.01 | tgt_test/bacc     | rand_rewind | tkt_rewind |        16 |    -0.0004692 |  -0.1207    |      5 | 0.1754    | 0.3507    |
| ids17→ddos19_d1     |   0.01 | tgt_test/bacc     | snip_tgt    | tkt_rewind |        16 |     0.001099  |   0.007746  |     13 | 0.002686  | 0.008057  |
| ids17→ddos19_d1     |   0.01 | tgt_test/bacc     | tkt_ft      | tkt_rewind |        16 |    -0.02054   |  -0.09951   |      0 | 3.052e-05 | 0.0001221 |
| ids17→ddos19_d1     |   0.01 | tgt_test/bacc     | tkt_reinit  | tkt_rewind |        16 |    -0.0005345 |  -0.004519  |      7 | 0.2312    | 0.3507    |
| ids17→ddos19_d1     |   0.01 | tgt_test/macro_f1 | rand_rewind | tkt_rewind |        16 |    -0.0005548 |  -0.1409    |      5 | 0.2114    | 0.2333    |
| ids17→ddos19_d1     |   0.01 | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |        16 |     0.001014  |   0.01012   |     13 | 0.002136  | 0.006409  |
| ids17→ddos19_d1     |   0.01 | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |        16 |    -0.03082   |  -0.1304    |      0 | 3.052e-05 | 0.0001221 |
| ids17→ddos19_d1     |   0.01 | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |        16 |    -0.0007828 |  -0.007564  |      6 | 0.1167    | 0.2333    |
| ids17→ddos19_d1     |   0.01 | flow/ESR_AUC      | rand_rewind | tkt_rewind |        16 |    -0.01272   |  -0.06896   |      4 | 0.03864   | 0.1159    |
| ids17→ddos19_d1     |   0.01 | flow/ESR_AUC      | snip_tgt    | tkt_rewind |        16 |    -0.01243   |  -0.04686   |      6 | 0.3484    | 0.5489    |
| ids17→ddos19_d1     |   0.01 | flow/ESR_AUC      | tkt_ft      | tkt_rewind |        16 |     0.6131    |   0.6193    |     16 | 3.052e-05 | 0.0001221 |
| ids17→ddos19_d1     |   0.01 | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |        16 |    -0.006384  |  -0.07406   |      8 | 0.2744    | 0.5489    |
| ids17→ddos19_d1     |   0.1  | tgt_test/bacc     | rand_rewind | tkt_rewind |         4 |     0.0001806 |  -0.1181    |      3 | 0.875     | 1         |
| ids17→ddos19_d1     |   0.1  | tgt_test/bacc     | snip_tgt    | tkt_rewind |         4 |     0.0004071 |   0.006011  |      3 | 0.25      | 0.75      |
| ids17→ddos19_d1     |   0.1  | tgt_test/bacc     | tkt_ft      | tkt_rewind |         4 |    -0.007924  |  -0.08672   |      0 | 0.125     | 0.5       |
| ids17→ddos19_d1     |   0.1  | tgt_test/bacc     | tkt_reinit  | tkt_rewind |         4 |     0.0001806 |   3.323e-06 |      3 | 0.875     | 1         |
| ids17→ddos19_d1     |   0.1  | tgt_test/macro_f1 | rand_rewind | tkt_rewind |         4 |     0.0001453 |  -0.1812    |      3 | 0.875     | 0.875     |
| ids17→ddos19_d1     |   0.1  | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |         4 |     0.0003629 |   0.0077    |      3 | 0.25      | 0.5       |
| ids17→ddos19_d1     |   0.1  | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |         4 |    -0.01209   |  -0.1139    |      0 | 0.125     | 0.5       |
| ids17→ddos19_d1     |   0.1  | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |         4 |     0.0001933 |   0.00038   |      4 | 0.125     | 0.5       |
| ids17→ddos19_d1     |   0.1  | flow/ESR_AUC      | rand_rewind | tkt_rewind |         3 |     0.193     |   0.09278   |      2 | 0.5       | 1         |
| ids17→ddos19_d1     |   0.1  | flow/ESR_AUC      | snip_tgt    | tkt_rewind |         4 |    -0.06315   |  -0.001256  |      2 | 1         | 1         |
| ids17→ddos19_d1     |   0.1  | flow/ESR_AUC      | tkt_ft      | tkt_rewind |         4 |     0.5747    |   0.5569    |      4 | 0.125     | 0.5       |
| ids17→ddos19_d1     |   0.1  | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |         4 |    -0.211     |  -0.2097    |      0 | 0.125     | 0.5       |
| ids17→ddos19_d1     |   1    | tgt_test/bacc     | rand_rewind | tkt_rewind |        16 |     0.0001096 |  -0.1214    |     10 | 0.7436    | 0.7436    |
| ids17→ddos19_d1     |   1    | tgt_test/bacc     | snip_tgt    | tkt_rewind |        16 |     0.0006135 |   0.004521  |     14 | 0.0001526 | 0.0004578 |
| ids17→ddos19_d1     |   1    | tgt_test/bacc     | tkt_ft      | tkt_rewind |        16 |    -0.001763  |  -0.01999   |      0 | 3.052e-05 | 0.0001221 |
| ids17→ddos19_d1     |   1    | tgt_test/bacc     | tkt_reinit  | tkt_rewind |        16 |     0.0002701 |   0.000173  |     11 | 0.2114    | 0.4229    |
| ids17→ddos19_d1     |   1    | tgt_test/macro_f1 | rand_rewind | tkt_rewind |        16 |     0.0001449 |  -0.1543    |     10 | 0.8603    | 0.8603    |
| ids17→ddos19_d1     |   1    | tgt_test/macro_f1 | snip_tgt    | tkt_rewind |        16 |     0.0005562 |   0.00551   |     15 | 9.155e-05 | 0.0002747 |
| ids17→ddos19_d1     |   1    | tgt_test/macro_f1 | tkt_ft      | tkt_rewind |        16 |    -0.002076  |  -0.02599   |      0 | 3.052e-05 | 0.0001221 |
| ids17→ddos19_d1     |   1    | tgt_test/macro_f1 | tkt_reinit  | tkt_rewind |        16 |     0.00029   |   0.0003183 |     11 | 0.2522    | 0.5045    |
| ids17→ddos19_d1     |   1    | flow/ESR_AUC      | rand_rewind | tkt_rewind |        15 |    -0.01777   |   0.02721   |      7 | 0.804     | 0.9908    |
| ids17→ddos19_d1     |   1    | flow/ESR_AUC      | snip_tgt    | tkt_rewind |        16 |     0.2065    |   0.1917    |     11 | 0.02496   | 0.07489   |
| ids17→ddos19_d1     |   1    | flow/ESR_AUC      | tkt_ft      | tkt_rewind |        16 |     0.3419    |   0.3413    |     15 | 6.104e-05 | 0.0002441 |
| ids17→ddos19_d1     |   1    | flow/ESR_AUC      | tkt_reinit  | tkt_rewind |        16 |     0.06478   |   0.07235   |     11 | 0.4954    | 0.9908    |