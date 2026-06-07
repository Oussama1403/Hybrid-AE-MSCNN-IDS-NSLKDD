# NSL-KDD Dataset Research Report

## Executive Summary
- KDDTrain+ samples: 125,973
- KDDTest+ samples: 22,544
- Zero-day / unknown attack types in test: 17
- Zero-day attack samples in test: 3,750

## Binary Traffic Split
    split  samples  normal  anomaly  normal_ratio  anomaly_ratio  normal_to_anomaly
KDDTrain+   125973   67343    58630      0.534583       0.465417           1.148610
 KDDTest+    22544    9711    12833      0.430758       0.569242           0.756721

## 5-Class Distribution
 class  train_count  test_count
Normal        67343        9711
   DoS        45927        7460
 Probe        11656        2421
   R2L          995        2885
   U2R           52          67

## Unknown Attack Labels
 attack_label  count_in_test
        mscan            996
      apache2            737
 processtable            685
    snmpguess            331
        saint            319
     mailbomb            293
snmpgetattack            178
   httptunnel            133
        named             17
           ps             15
     sendmail             14
        xterm             13
        xlock              9
       xsnoop              4
    sqlattack              2
     udpstorm              2
         worm              2

## KDDTest+ Known vs Unknown Label Summary
                  label_status  sample_count  sample_share  sample_share_pct
   Known (exists in KDDTrain+)         18794      0.833659         83.365862
Unknown (zero-day in KDDTest+)          3750      0.166341         16.634138
                         Total         22544      1.000000        100.000000

## KDDTest+ Label Table
          label  count_in_test  status  is_unknown_in_train  percent_of_test
          mscan            996 Unknown                    1         4.418027
        apache2            737 Unknown                    1         3.269163
   processtable            685 Unknown                    1         3.038502
      snmpguess            331 Unknown                    1         1.468240
          saint            319 Unknown                    1         1.415011
       mailbomb            293 Unknown                    1         1.299681
  snmpgetattack            178 Unknown                    1         0.789567
     httptunnel            133 Unknown                    1         0.589957
          named             17 Unknown                    1         0.075408
             ps             15 Unknown                    1         0.066537
       sendmail             14 Unknown                    1         0.062101
          xterm             13 Unknown                    1         0.057665
          xlock              9 Unknown                    1         0.039922
         xsnoop              4 Unknown                    1         0.017743
      sqlattack              2 Unknown                    1         0.008872
       udpstorm              2 Unknown                    1         0.008872
           worm              2 Unknown                    1         0.008872
         normal           9711   Known                    0        43.075763
        neptune           4657   Known                    0        20.657381
   guess_passwd           1231   Known                    0         5.460433
    warezmaster            944   Known                    0         4.187367
          satan            735   Known                    0         3.260291
          smurf            665   Known                    0         2.949787
           back            359   Known                    0         1.592441
      portsweep            157   Known                    0         0.696416
        ipsweep            141   Known                    0         0.625444
           nmap             73   Known                    0         0.323811
            pod             41   Known                    0         0.181867
buffer_overflow             20   Known                    0         0.088715
       multihop             18   Known                    0         0.079844
        rootkit             13   Known                    0         0.057665
       teardrop             12   Known                    0         0.053229
           land              7   Known                    0         0.031050
      ftp_write              3   Known                    0         0.013307
     loadmodule              2   Known                    0         0.008872
           perl              2   Known                    0         0.008872
            phf              2   Known                    0         0.008872
           imap              1   Known                    0         0.004436

## Per-Attack Label Train vs Test (Top Rows)
 attack_label  train_count  test_count  is_unknown_in_train
        mscan            0         996                    1
      apache2            0         737                    1
 processtable            0         685                    1
    snmpguess            0         331                    1
        saint            0         319                    1
     mailbomb            0         293                    1
snmpgetattack            0         178                    1
   httptunnel            0         133                    1
        named            0          17                    1
           ps            0          15                    1
     sendmail            0          14                    1
        xterm            0          13                    1
        xlock            0           9                    1
       xsnoop            0           4                    1
    sqlattack            0           2                    1
     udpstorm            0           2                    1
         worm            0           2                    1
       normal        67343        9711                    0
      neptune        41214        4657                    0
 guess_passwd           53        1231                    0

## Top Numerical Feature Shifts
                 feature  standardized_mean_shift  variance_ratio_test_over_train
               src_bytes                -0.008446                        0.006486
               dst_bytes                -0.006233                        0.000028
                duration                -0.032622                        0.291896
      dst_host_srv_count                 0.225610                        1.019592
          dst_host_count                 0.121262                        0.898445
                   count                -0.041727                        1.260026
               srv_count                 0.041673                        1.503392
                num_root                -0.010323                        0.108619
    dst_host_serror_rate                -0.505694                        0.377096
             serror_rate                -0.479657                        0.437674
dst_host_srv_serror_rate                -0.480217                        0.399986
         srv_serror_rate                -0.470638                        0.445373

## Suggested Presentation Angle
- Emphasize the strict train/test protocol and the presence of unseen attacks in KDDTest+.
- Use the 5-class table to show how the benchmark compresses many labels into Normal, DoS, Probe, R2L, and U2R.
- Use the numerical drift plots to argue that the test distribution is not trivial and is a meaningful generalization challenge.