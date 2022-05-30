import streamlit as st
from joblib import dump, load
import pandas as pd
import numpy as np

log_model = load('university_admission.joblib')

def classify(gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research):
    inputs=np.array([gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research])
    
    data = [[337.  , 118.  ,   4.  ,   4.5 ,   4.5 ,   9.65,   1.  ],[324.  , 107.  ,   4.  ,   4.  ,   4.5 ,   8.87,   1. ],[316.  , 104.  ,   3.  ,   3.  ,   3.5 ,   8.  ,   1.  ],
       [322.  , 110.  ,   3.  ,   3.5 ,   2.5 ,   8.67,   1.  ],[330.  , 115.  ,   5.  ,   4.5 ,   3.  ,   9.34,   1.  ],
       [321.  , 109.  ,   3.  ,   3.  ,   4.  ,   8.2 ,   1.  ],[302. , 102. ,   1. ,   2. ,   1.5,   8. ,   0. ],
       [323. , 108. ,   3. ,   3.5,   3. ,   8.6,   0. ],
       [325. , 106. ,   3. ,   3.5,   4. ,   8.4,   1. ],[328. , 112. ,   4. ,   4. ,   4.5,   9.1,   1. ],
       [307. , 109. ,   3. ,   4. ,   3. ,   8. ,   1. ],
       [311. , 104. ,   3. ,   3.5,   2. ,   8.2,   1. ],
       [314. , 105. ,   3. ,   3.5,   2.5,   8.3,   0. ],[319. , 106. ,   3. ,   4. ,   3. ,   8. ,   1. ],
       [318. , 110. ,   3. ,   4. ,   3. ,   8.8,   0. ],
       [303. , 102. ,   3. ,   3.5,   3. ,   8.5,   0. ],
       [312. , 107. ,   3. ,   3. ,   2. ,   7.9,   1. ],
       [325. , 114. ,   4. ,   3. ,   2. ,   8.4,   0. ],
       [328. , 116. ,   5. ,   5. ,   5. ,   9.5,   1. ],
       [334. , 119. ,   5. ,   5. ,   4.5,   9.7,   1. ],
       [336. , 119. ,   5. ,   4. ,   3.5,   9.8,   1. ],[322. , 109. ,   5. ,   4.5,   3.5,   8.8,   0. ],
       [298. ,  98. ,   2. ,   1.5,   2.5,   7.5,   1. ],
       [295. ,  93. ,   1. ,   2. ,   2. ,   7.2,   0. ],
       [310. ,  99. ,   2. ,   1.5,   2. ,   7.3,   0. ],
       [300. ,  97. ,   2. ,   3. ,   3. ,   8.1,   1. ],
       [327. , 103. ,   3. ,   4. ,   4. ,   8.3,   1. ],
       [338. , 118. ,   4. ,   3. ,   4.5,   9.4,   1. ],
       [340. , 114. ,   5. ,   4. ,   4. ,   9.6,   1. ],
       [331. , 112. ,   5. ,   4. ,   5. ,   9.8,   1. ],
       [320. , 110. ,   5. ,   5. ,   5. ,   9.2,   1. ],
       [299. , 106. ,   2. ,   4. ,   4. ,   8.4,   0. ],
       [300. , 105. ,   1. ,   1. ,   2. ,   7.8,   0. ],
       [304. , 105. ,   1. ,   3. ,   1.5,   7.5,   0. ],
       [307. , 108. ,   2. ,   4. ,   3.5,   7.7,   0. ],[316.  , 105.  ,   2.  ,   2.5 ,   2.5 ,   8.2 ,   1.  ],
       [313.  , 107.  ,   2.  ,   2.5 ,   2.  ,   8.5 ,   1.  ],
       [332.  , 117.  ,   4.  ,   4.5 ,   4.  ,   9.1 ,   0.  ],
       [326.  , 113.  ,   5.  ,   4.5 ,   4.  ,   9.4 ,   1.  ],
       [322.  , 110.  ,   5.  ,   5.  ,   4.  ,   9.1 ,   1.  ],
       [329.  , 114.  ,   5.  ,   4.  ,   5.  ,   9.3 ,   1.  ],
       [339.  , 119.  ,   5.  ,   4.5 ,   4.  ,   9.7 ,   0.  ],
       [321.  , 110.  ,   3.  ,   3.5 ,   5.  ,   8.85,   1.  ],
       [327.  , 111.  ,   4.  ,   3.  ,   4.  ,   8.4 ,   1.  ],
       [313.  ,  98.  ,   3.  ,   2.5 ,   4.5 ,   8.3 ,   1.  ],
       [312.  , 100.  ,   2.  ,   1.5 ,   3.5 ,   7.9 ,   1.  ],
       [334.  , 116.  ,   4.  ,   4.  ,   3.  ,   8.  ,   1.  ],
       [324.  , 112.  ,   4.  ,   4.  ,   2.5 ,   8.1 ,   1.  ],
       [322.  , 110.  ,   3.  ,   3.  ,   3.5 ,   8.  ,   0.  ],
       [320.  , 103.  ,   3.  ,   3.  ,   3.  ,   7.7 ,   0.  ],
       [316.  , 102.  ,   3.  ,   2.  ,   3.  ,   7.4 ,   0.  ],
       [298.  ,  99.  ,   2.  ,   4.  ,   2.  ,   7.6 ,   0.  ],
       [300.  ,  99.  ,   1.  ,   3.  ,   2.  ,   6.8 ,   1.  ],
       [311.  , 104.  ,   2.  ,   2.  ,   2.  ,   8.3 ,   0.  ],[307.  , 101.  ,   3.  ,   4.  ,   3.  ,   8.2 ,   0.  ],
       [304.  , 105.  ,   2.  ,   3.  ,   3.  ,   8.2 ,   1.  ],
       [315.  , 107.  ,   2.  ,   4.  ,   3.  ,   8.5 ,   1.  ],
       [325.  , 111.  ,   3.  ,   3.  ,   3.5 ,   8.7 ,   0.  ],
       [325.  , 112.  ,   4.  ,   3.5 ,   3.5 ,   8.92,   0.  ],
       [327.  , 114.  ,   3.  ,   3.  ,   3.  ,   9.02,   0.  ],
       [316.  , 107.  ,   2.  ,   3.5 ,   3.5 ,   8.64,   1.  ],
       [318.  , 109.  ,   3.  ,   3.5 ,   4.  ,   9.22,   1.  ],
       [328.  , 115.  ,   4.  ,   4.5 ,   4.  ,   9.16,   1.  ],
       [332.  , 118.  ,   5.  ,   5.  ,   5.  ,   9.64,   1.  ],
       [336.  , 112.  ,   5.  ,   5.  ,   5.  ,   9.76,   1.  ],
       [321.  , 111.  ,   5.  ,   5.  ,   5.  ,   9.45,   1.  ],
       [314.  , 108.  ,   4.  ,   4.5 ,   4.  ,   9.04,   1.  ],
       [314.  , 106.  ,   3.  ,   3.  ,   5.  ,   8.9 ,   0.  ],
       [329.  , 114.  ,   2.  ,   2.  ,   4.  ,   8.56,   1.  ],
       [327.  , 112.  ,   3.  ,   3.  ,   3.  ,   8.72,   1.  ],
       [301.  ,  99.  ,   2.  ,   3.  ,   2.  ,   8.22,   0.  ],
       [296.  ,  95.  ,   2.  ,   3.  ,   2.  ,   7.54,   1.  ],
       [294.  ,  93.  ,   1.  ,   1.5 ,   2.  ,   7.36,   0.  ],[340.  , 120.  ,   4.  ,   5.  ,   5.  ,   9.5 ,   1.  ],
       [320.  , 110.  ,   5.  ,   5.  ,   4.5 ,   9.22,   1.  ],
       [322.  , 115.  ,   5.  ,   4.  ,   4.5 ,   9.36,   1.  ],
       [340.  , 115.  ,   5.  ,   4.5 ,   4.5 ,   9.45,   1.  ],
       [319.  , 103.  ,   4.  ,   4.5 ,   3.5 ,   8.66,   0.  ],
       [315.  , 106.  ,   3.  ,   4.5 ,   3.5 ,   8.42,   0.  ],
       [317.  , 107.  ,   2.  ,   3.5 ,   3.  ,   8.28,   0.  ],
       [314.  , 108.  ,   3.  ,   4.5 ,   3.5 ,   8.14,   0.  ],
       [316.  , 109.  ,   4.  ,   4.5 ,   3.5 ,   8.76,   1.  ],
       [318.  , 106.  ,   2.  ,   4.  ,   4.  ,   7.92,   1.  ],
       [299.  ,  97.  ,   3.  ,   5.  ,   3.5 ,   7.66,   0.  ],
       [298.  ,  98.  ,   2.  ,   4.  ,   3.  ,   8.03,   0.  ],
       [301.  ,  97.  ,   2.  ,   3.  ,   3.  ,   7.88,   1.  ],
       [303.  ,  99.  ,   3.  ,   2.  ,   2.5 ,   7.66,   0.  ],
       [304.  , 100.  ,   4.  ,   1.5 ,   2.5 ,   7.84,   0.  ],
       [306.  , 100.  ,   2.  ,   3.  ,   3.  ,   8.  ,   0.  ],
       [331.  , 120.  ,   3.  ,   4.  ,   4.  ,   8.96,   1.  ],
       [332.  , 119.  ,   4.  ,   5.  ,   4.5 ,   9.24,   1.  ],
       [323.  , 113.  ,   3.  ,   4.  ,   4.  ,   8.88,   1.  ],[312.  , 105.  ,   2.  ,   2.5 ,   3.  ,   8.12,   0.  ],
       [314.  , 106.  ,   2.  ,   4.  ,   3.5 ,   8.25,   0.  ],
       [317.  , 104.  ,   2.  ,   4.5 ,   4.  ,   8.47,   0.  ],
       [326.  , 112.  ,   3.  ,   3.5 ,   3.  ,   9.05,   1.  ],
       [316.  , 110.  ,   3.  ,   4.  ,   4.5 ,   8.78,   1.  ],
       [329.  , 111.  ,   4.  ,   4.5 ,   4.5 ,   9.18,   1.  ],
       [338.  , 117.  ,   4.  ,   3.5 ,   4.5 ,   9.46,   1.  ],
       [331.  , 116.  ,   5.  ,   5.  ,   5.  ,   9.38,   1.  ],
       [304.  , 103.  ,   5.  ,   5.  ,   4.  ,   8.64,   0.  ],
       [305.  , 108.  ,   5.  ,   3.  ,   3.  ,   8.48,   0.  ],
       [321.  , 109.  ,   4.  ,   4.  ,   4.  ,   8.68,   1.  ],
       [301.  , 107.  ,   3.  ,   3.5 ,   3.5 ,   8.34,   1.  ],
       [320.  , 110.  ,   2.  ,   4.  ,   3.5 ,   8.56,   0.  ],
       [311.  , 105.  ,   3.  ,   3.5 ,   3.  ,   8.45,   1.  ],
       [310.  , 106.  ,   4.  ,   4.5 ,   4.5 ,   9.04,   1.  ],
       [299.  , 102.  ,   3.  ,   4.  ,   3.5 ,   8.62,   0.  ],
       [290.  , 104.  ,   4.  ,   2.  ,   2.5 ,   7.46,   0.  ],
       [296.  ,  99.  ,   2.  ,   3.  ,   3.5 ,   7.28,   0.  ],
       [327.  , 104.  ,   5.  ,   3.  ,   3.5 ,   8.84,   1.  ],
       [335.  , 117.  ,   5.  ,   5.  ,   5.  ,   9.56,   1.  ],
       [334.  , 119.  ,   5.  ,   4.5 ,   4.5 ,   9.48,   1.  ],
       [310.  , 106.  ,   4.  ,   1.5 ,   2.5 ,   8.36,   0.  ],
       [308.  , 108.  ,   3.  ,   3.5 ,   3.5 ,   8.22,   0.  ],
       [301.  , 106.  ,   4.  ,   2.5 ,   3.  ,   8.47,   0.  ],[323.  , 113.  ,   3.  ,   4.  ,   3.  ,   9.32,   1.  ],
       [319.  , 112.  ,   3.  ,   2.5 ,   2.  ,   8.71,   1.  ],
       [326.  , 112.  ,   3.  ,   3.5 ,   3.  ,   9.1 ,   1.  ],
       [333.  , 118.  ,   5.  ,   5.  ,   5.  ,   9.35,   1.  ],
       [339.  , 114.  ,   5.  ,   4.  ,   4.5 ,   9.76,   1.  ],
       [303.  , 105.  ,   5.  ,   5.  ,   4.5 ,   8.65,   0.  ],
       [309.  , 105.  ,   5.  ,   3.5 ,   3.5 ,   8.56,   0.  ],
       [323.  , 112.  ,   5.  ,   4.  ,   4.5 ,   8.78,   0.  ],
       [333.  , 113.  ,   5.  ,   4.  ,   4.  ,   9.28,   1.  ],
       [314.  , 109.  ,   4.  ,   3.5 ,   4.  ,   8.77,   1.  ],
       [312.  , 103.  ,   3.  ,   5.  ,   4.  ,   8.45,   0.  ],
       [316.  , 100.  ,   2.  ,   1.5 ,   3.  ,   8.16,   1.  ],
       [326.  , 116.  ,   2.  ,   4.5 ,   3.  ,   9.08,   1.  ],
       [318.  , 109.  ,   1.  ,   3.5 ,   3.5 ,   9.12,   0.  ],
       [329.  , 110.  ,   2.  ,   4.  ,   3.  ,   9.15,   1.  ],
       [332.  , 118.  ,   2.  ,   4.5 ,   3.5 ,   9.36,   1.  ],
       [331.  , 115.  ,   5.  ,   4.  ,   3.5 ,   9.44,   1.  ],
       [340.  , 120.  ,   4.  ,   4.5 ,   4.  ,   9.92,   1.  ],
       [325.  , 112.  ,   2.  ,   3.  ,   3.5 ,   8.96,   1.  ],
       [320.  , 113.  ,   2.  ,   2.  ,   2.5 ,   8.64,   1.  ],
       [315.  , 105.  ,   3.  ,   2.  ,   2.5 ,   8.48,   0.  ],
       [326.  , 114.  ,   3.  ,   3.  ,   3.  ,   9.11,   1.  ],
       [339.  , 116.  ,   4.  ,   4.  ,   3.5 ,   9.8 ,   1.  ],
       [311.  , 106.  ,   2.  ,   3.5 ,   3.  ,   8.26,   1.  ],[332.  , 116.  ,   5.  ,   5.  ,   5.  ,   9.28,   1.  ],
       [321.  , 112.  ,   5.  ,   5.  ,   5.  ,   9.06,   1.  ],
       [324.  , 105.  ,   3.  ,   3.  ,   4.  ,   8.75,   0.  ],
       [326.  , 108.  ,   3.  ,   3.  ,   3.5 ,   8.89,   0.  ],
       [312.  , 109.  ,   3.  ,   3.  ,   3.  ,   8.69,   0.  ],
       [315.  , 105.  ,   3.  ,   2.  ,   2.5 ,   8.34,   0.  ],
       [309.  , 104.  ,   2.  ,   2.  ,   2.5 ,   8.26,   0.  ],
       [306.  , 106.  ,   2.  ,   2.  ,   2.5 ,   8.14,   0.  ],
       [297.  , 100.  ,   1.  ,   1.5 ,   2.  ,   7.9 ,   0.  ],
       [315.  , 103.  ,   1.  ,   1.5 ,   2.  ,   7.86,   0.  ],
       [298.  ,  99.  ,   1.  ,   1.5 ,   3.  ,   7.46,   0.  ],
       [318.  , 109.  ,   3.  ,   3.  ,   3.  ,   8.5 ,   0.  ],
       [317.  , 105.  ,   3.  ,   3.5 ,   3.  ,   8.56,   0.  ],
       [329.  , 111.  ,   4.  ,   4.5 ,   4.  ,   9.01,   1.  ],
       [322.  , 110.  ,   5.  ,   4.5 ,   4.  ,   8.97,   0.  ],
       [302.  , 102.  ,   3.  ,   3.5 ,   5.  ,   8.33,   0.  ],
       [313.  , 102.  ,   3.  ,   2.  ,   3.  ,   8.27,   0.  ],
       [293.  ,  97.  ,   2.  ,   2.  ,   4.  ,   7.8 ,   1.  ],
       [311.  ,  99.  ,   2.  ,   2.5 ,   3.  ,   7.98,   0.  ],
       [312.  , 101.  ,   2.  ,   2.5 ,   3.5 ,   8.04,   1.  ],
       [334.  , 117.  ,   5.  ,   4.  ,   4.5 ,   9.07,   1.  ],
       [322.  , 110.  ,   4.  ,   4.  ,   5.  ,   9.13,   1.  ],
       [323.  , 113.  ,   4.  ,   4.  ,   4.5 ,   9.23,   1.  ],
       [321.  , 111.  ,   4.  ,   4.  ,   4.  ,   8.97,   1.  ],
       [320.  , 111.  ,   4.  ,   4.5 ,   3.5 ,   8.87,   1.  ],
       [329.  , 119.  ,   4.  ,   4.5 ,   4.5 ,   9.16,   1.  ],
       [319.  , 110.  ,   3.  ,   3.5 ,   3.5 ,   9.04,   0.  ],
       [309.  , 108.  ,   3.  ,   2.5 ,   3.  ,   8.12,   0.  ],
       [307.  , 102.  ,   3.  ,   3.  ,   3.  ,   8.27,   0.  ],[305.  , 107.  ,   2.  ,   2.5 ,   2.5 ,   8.42,   0.  ],
       [299.  , 100.  ,   2.  ,   3.  ,   3.5 ,   7.88,   0.  ],
       [314.  , 110.  ,   3.  ,   4.  ,   4.  ,   8.8 ,   0.  ],
       [316.  , 106.  ,   2.  ,   2.5 ,   4.  ,   8.32,   0.  ],
       [327.  , 113.  ,   4.  ,   4.5 ,   4.5 ,   9.11,   1.  ],
       [317.  , 107.  ,   3.  ,   3.5 ,   3.  ,   8.68,   1.  ],
       [335.  , 118.  ,   5.  ,   4.5 ,   3.5 ,   9.44,   1.  ],
       [331.  , 115.  ,   5.  ,   4.5 ,   3.5 ,   9.36,   1.  ],
       [324.  , 112.  ,   5.  ,   5.  ,   5.  ,   9.08,   1.  ],
       [324.  , 111.  ,   5.  ,   4.5 ,   4.  ,   9.16,   1.  ],
       [323.  , 110.  ,   5.  ,   4.  ,   5.  ,   8.98,   1.  ],
       [322.  , 114.  ,   5.  ,   4.5 ,   4.  ,   8.94,   1.  ],
       [336.  , 118.  ,   5.  ,   4.5 ,   5.  ,   9.53,   1.  ],
       [316.  , 109.  ,   3.  ,   3.5 ,   3.  ,   8.76,   0.  ],
       [307.  , 107.  ,   2.  ,   3.  ,   3.5 ,   8.52,   1.  ],
       [306.  , 105.  ,   2.  ,   3.  ,   2.5 ,   8.26,   0.  ],
       [310.  , 106.  ,   2.  ,   3.5 ,   2.5 ,   8.33,   0.  ],
       [311.  , 104.  ,   3.  ,   4.5 ,   4.5 ,   8.43,   0.  ],
       [313.  , 107.  ,   3.  ,   4.  ,   4.5 ,   8.69,   0.  ],[315.  , 110.  ,   2.  ,   3.5 ,   3.  ,   8.46,   1.  ],
       [340.  , 120.  ,   5.  ,   4.5 ,   4.5 ,   9.91,   1.  ],
       [334.  , 120.  ,   5.  ,   4.  ,   5.  ,   9.87,   1.  ],
       [298.  , 105.  ,   3.  ,   3.5 ,   4.  ,   8.54,   0.  ],
       [295.  ,  99.  ,   2.  ,   2.5 ,   3.  ,   7.65,   0.  ],
       [315.  ,  99.  ,   2.  ,   3.5 ,   3.  ,   7.89,   0.  ],
       [310.  , 102.  ,   3.  ,   3.5 ,   4.  ,   8.02,   1.  ],
       [305.  , 106.  ,   2.  ,   3.  ,   3.  ,   8.16,   0.  ],
       [301.  , 104.  ,   3.  ,   3.5 ,   4.  ,   8.12,   1.  ],
       [325.  , 108.  ,   4.  ,   4.5 ,   4.  ,   9.06,   1.  ],
       [328.  , 110.  ,   4.  ,   5.  ,   4.  ,   9.14,   1.  ],
       [338.  , 120.  ,   4.  ,   5.  ,   5.  ,   9.66,   1.  ],
       [333.  , 119.  ,   5.  ,   5.  ,   4.5 ,   9.78,   1.  ],
       [331.  , 117.  ,   4.  ,   4.5 ,   5.  ,   9.42,   1.  ],
       [330.  , 116.  ,   5.  ,   5.  ,   4.5 ,   9.36,   1.  ],
       [322.  , 112.  ,   4.  ,   4.5 ,   4.5 ,   9.26,   1.  ],
       [321.  , 109.  ,   4.  ,   4.  ,   4.  ,   9.13,   1.  ],
       [324.  , 110.  ,   4.  ,   3.  ,   3.5 ,   8.97,   1.  ],
       [312.  , 104.  ,   3.  ,   3.5 ,   3.5 ,   8.42,   0.  ],
       [313.  , 103.  ,   3.  ,   4.  ,   4.  ,   8.75,   0.  ],
       [316.  , 110.  ,   3.  ,   3.5 ,   4.  ,   8.56,   0.  ],
       [324.  , 113.  ,   4.  ,   4.5 ,   4.  ,   8.79,   0.  ],
       [308.  , 109.  ,   2.  ,   3.  ,   4.  ,   8.45,   0.  ],
       [305.  , 105.  ,   2.  ,   3.  ,   2.  ,   8.23,   0.  ],
       [296.  ,  99.  ,   2.  ,   2.5 ,   2.5 ,   8.03,   0.  ],
       [306.  , 110.  ,   2.  ,   3.5 ,   4.  ,   8.45,   0.  ],
       [312.  , 110.  ,   2.  ,   3.5 ,   3.  ,   8.53,   0.  ],
       [318.  , 112.  ,   3.  ,   4.  ,   3.5 ,   8.67,   0.  ],
       [324.  , 111.  ,   4.  ,   3.  ,   3.  ,   9.01,   1.  ],[319.  , 106.  ,   3.  ,   3.5 ,   2.5 ,   8.33,   1.  ],
       [312.  , 107.  ,   2.  ,   2.5 ,   3.5 ,   8.27,   0.  ],
       [304.  , 100.  ,   2.  ,   2.5 ,   3.5 ,   8.07,   0.  ],
       [330.  , 113.  ,   5.  ,   5.  ,   4.  ,   9.31,   1.  ],
       [326.  , 111.  ,   5.  ,   4.5 ,   4.  ,   9.23,   1.  ],
       [325.  , 112.  ,   4.  ,   4.  ,   4.5 ,   9.17,   1.  ],
       [329.  , 114.  ,   5.  ,   4.5 ,   5.  ,   9.19,   1.  ],
       [310.  , 104.  ,   3.  ,   2.  ,   3.5 ,   8.37,   0.  ],
       [299.  , 100.  ,   1.  ,   1.5 ,   2.  ,   7.89,   0.  ],
       [296.  , 101.  ,   1.  ,   2.5 ,   3.  ,   7.68,   0.  ],
       [317.  , 103.  ,   2.  ,   2.5 ,   2.  ,   8.15,   0.  ],
       [324.  , 115.  ,   3.  ,   3.5 ,   3.  ,   8.76,   1.  ],
       [325.  , 114.  ,   3.  ,   3.5 ,   3.  ,   9.04,   1.  ],
       [314.  , 107.  ,   2.  ,   2.5 ,   4.  ,   8.56,   0.  ],
       [328.  , 110.  ,   4.  ,   4.  ,   2.5 ,   9.02,   1.  ],
       [316.  , 105.  ,   3.  ,   3.  ,   3.5 ,   8.73,   0.  ],
       [311.  , 104.  ,   2.  ,   2.5 ,   3.5 ,   8.48,   0.  ],
       [324.  , 110.  ,   3.  ,   3.5 ,   4.  ,   8.87,   1.  ],
       [321.  , 111.  ,   3.  ,   3.5 ,   4.  ,   8.83,   1.  ],
       [320.  , 104.  ,   3.  ,   3.  ,   2.5 ,   8.57,   1.  ],
       [316.  ,  99.  ,   2.  ,   2.5 ,   3.  ,   9.  ,   0.  ],
       [318.  , 100.  ,   2.  ,   2.5 ,   3.5 ,   8.54,   1.  ],
       [335.  , 115.  ,   4.  ,   4.5 ,   4.5 ,   9.68,   1.  ],
       [321.  , 114.  ,   4.  ,   4.  ,   5.  ,   9.12,   0.  ],
       [307.  , 110.  ,   4.  ,   4.  ,   4.5 ,   8.37,   0.  ],
       [309.  ,  99.  ,   3.  ,   4.  ,   4.  ,   8.56,   0.  ],
       [324.  , 100.  ,   3.  ,   4.  ,   5.  ,   8.64,   1.  ],
       [326.  , 102.  ,   4.  ,   5.  ,   5.  ,   8.76,   1.  ],
       [331.  , 119.  ,   4.  ,   5.  ,   4.5 ,   9.34,   1.  ],
       [327.  , 108.  ,   5.  ,   5.  ,   3.5 ,   9.13,   1.  ],
       [312.  , 104.  ,   3.  ,   3.5 ,   4.  ,   8.09,   0.  ],
       [308.  , 103.  ,   2.  ,   2.5 ,   4.  ,   8.36,   1.  ],
       [324.  , 111.  ,   3.  ,   2.5 ,   1.5 ,   8.79,   1.  ],
       [325.  , 110.  ,   2.  ,   3.  ,   2.5 ,   8.76,   1.  ],
       [313.  , 102.  ,   3.  ,   2.5 ,   2.5 ,   8.68,   0.  ],
       [312.  , 105.  ,   2.  ,   2.  ,   2.5 ,   8.45,   0.  ],
       [314.  , 107.  ,   3.  ,   3.  ,   3.5 ,   8.17,   1.  ],
       [327.  , 113.  ,   4.  ,   4.5 ,   5.  ,   9.14,   0.  ],
       [308.  , 108.  ,   4.  ,   4.5 ,   5.  ,   8.34,   0.  ],[299.  ,  96.  ,   2.  ,   1.5 ,   2.  ,   7.86,   0.  ],
       [294.  ,  95.  ,   1.  ,   1.5 ,   1.5 ,   7.64,   0.  ],
       [312.  ,  99.  ,   1.  ,   1.  ,   1.5 ,   8.01,   1.  ],
       [315.  , 100.  ,   1.  ,   2.  ,   2.5 ,   7.95,   0.  ],
       [322.  , 110.  ,   3.  ,   3.5 ,   3.  ,   8.96,   1.  ],
       [329.  , 113.  ,   5.  ,   5.  ,   4.5 ,   9.45,   1.  ],
       [320.  , 101.  ,   2.  ,   2.5 ,   3.  ,   8.62,   0.  ],
       [308.  , 103.  ,   2.  ,   3.  ,   3.5 ,   8.49,   0.  ],
       [304.  , 102.  ,   2.  ,   3.  ,   4.  ,   8.73,   0.  ],
       [311.  , 102.  ,   3.  ,   4.5 ,   4.  ,   8.64,   1.  ],
       [317.  , 110.  ,   3.  ,   4.  ,   4.5 ,   9.11,   1.  ],
       [312.  , 106.  ,   3.  ,   4.  ,   3.5 ,   8.79,   1.  ],
       [321.  , 111.  ,   3.  ,   2.5 ,   3.  ,   8.9 ,   1.  ],
       [340.  , 112.  ,   4.  ,   5.  ,   4.5 ,   9.66,   1.  ],
       [331.  , 116.  ,   5.  ,   4.  ,   4.  ,   9.26,   1.  ],
       [336.  , 118.  ,   5.  ,   4.5 ,   4.  ,   9.19,   1.  ],
       [324.  , 114.  ,   5.  ,   5.  ,   4.5 ,   9.08,   1.  ],
       [314.  , 104.  ,   4.  ,   5.  ,   5.  ,   9.02,   0.  ],
       [313.  , 109.  ,   3.  ,   4.  ,   3.5 ,   9.  ,   0.  ],
       [307.  , 105.  ,   2.  ,   2.5 ,   3.  ,   7.65,   0.  ],
       [300.  , 102.  ,   2.  ,   1.5 ,   2.  ,   7.87,   0.  ],
       [302.  ,  99.  ,   2.  ,   1.  ,   2.  ,   7.97,   0.  ],
       [312.  ,  98.  ,   1.  ,   3.5 ,   3.  ,   8.18,   1.  ],
       [316.  , 101.  ,   2.  ,   2.5 ,   2.  ,   8.32,   1.  ],
       [317.  , 100.  ,   2.  ,   3.  ,   2.5 ,   8.57,   0.  ],
       [310.  , 107.  ,   3.  ,   3.5 ,   3.5 ,   8.67,   0.  ],
       [320.  , 120.  ,   3.  ,   4.  ,   4.5 ,   9.11,   0.  ],
       [330.  , 114.  ,   3.  ,   4.5 ,   4.5 ,   9.24,   1.  ],
       [305.  , 112.  ,   3.  ,   3.  ,   3.5 ,   8.65,   0.  ],[319.  , 108.  ,   2.  ,   2.5 ,   3.  ,   8.76,   0.  ],
       [322.  , 105.  ,   2.  ,   3.  ,   3.  ,   8.45,   1.  ],
       [323.  , 107.  ,   3.  ,   3.5 ,   3.5 ,   8.55,   1.  ],
       [313.  , 106.  ,   2.  ,   2.5 ,   2.  ,   8.43,   0.  ],
       [321.  , 109.  ,   3.  ,   3.5 ,   3.5 ,   8.8 ,   1.  ],
       [323.  , 110.  ,   3.  ,   4.  ,   3.5 ,   9.1 ,   1.  ],
       [325.  , 112.  ,   4.  ,   4.  ,   4.  ,   9.  ,   1.  ],
       [312.  , 108.  ,   3.  ,   3.5 ,   3.  ,   8.53,   0.  ],
       [308.  , 110.  ,   4.  ,   3.5 ,   3.  ,   8.6 ,   0.  ],
       [320.  , 104.  ,   3.  ,   3.  ,   3.5 ,   8.74,   1.  ],
       [328.  , 108.  ,   4.  ,   4.5 ,   4.  ,   9.18,   1.  ],
       [311.  , 107.  ,   4.  ,   4.5 ,   4.5 ,   9.  ,   1.  ],
       [301.  , 100.  ,   3.  ,   3.5 ,   3.  ,   8.04,   0.  ],
       [305.  , 105.  ,   2.  ,   3.  ,   4.  ,   8.13,   0.  ],
       [308.  , 104.  ,   2.  ,   2.5 ,   3.  ,   8.07,   0.  ],
       [298.  , 101.  ,   2.  ,   1.5 ,   2.  ,   7.86,   0.  ],
       [300.  ,  99.  ,   1.  ,   1.  ,   2.5 ,   8.01,   0.  ],
       [324.  , 111.  ,   3.  ,   2.5 ,   2.  ,   8.8 ,   1.  ],
       [327.  , 113.  ,   4.  ,   3.5 ,   3.  ,   8.69,   1.  ],
       [317.  , 106.  ,   3.  ,   4.  ,   3.5 ,   8.5 ,   1.  ],
       [323.  , 104.  ,   3.  ,   4.  ,   4.  ,   8.44,   1.  ],
       [314.  , 107.  ,   2.  ,   2.5 ,   4.  ,   8.27,   0.  ],
       [305.  , 102.  ,   2.  ,   2.  ,   2.5 ,   8.18,   0.  ],
       [315.  , 104.  ,   3.  ,   3.  ,   2.5 ,   8.33,   0.  ],
       [326.  , 116.  ,   3.  ,   3.5 ,   4.  ,   9.14,   1.  ],
       [299.  , 100.  ,   3.  ,   2.  ,   2.  ,   8.02,   0.  ],
       [295.  , 101.  ,   2.  ,   2.5 ,   2.  ,   7.86,   0.  ],
       [324.  , 112.  ,   4.  ,   4.  ,   3.5 ,   8.77,   1.  ],
       [297.  ,  96.  ,   2.  ,   2.5 ,   1.5 ,   7.89,   0.  ],[311.  , 105.  ,   2.  ,   3.  ,   2.  ,   8.12,   1.  ],
       [308.  , 106.  ,   3.  ,   3.5 ,   2.5 ,   8.21,   1.  ],
       [319.  , 108.  ,   3.  ,   3.  ,   3.5 ,   8.54,   1.  ],
       [312.  , 107.  ,   4.  ,   4.5 ,   4.  ,   8.65,   1.  ],
       [325.  , 111.  ,   4.  ,   4.  ,   4.5 ,   9.11,   1.  ],
       [319.  , 110.  ,   3.  ,   3.  ,   2.5 ,   8.79,   0.  ],
       [332.  , 118.  ,   5.  ,   5.  ,   5.  ,   9.47,   1.  ],
       [323.  , 108.  ,   5.  ,   4.  ,   4.  ,   8.74,   1.  ],
       [324.  , 107.  ,   5.  ,   3.5 ,   4.  ,   8.66,   1.  ],
       [312.  , 107.  ,   3.  ,   3.  ,   3.  ,   8.46,   1.  ],
       [326.  , 110.  ,   3.  ,   3.5 ,   3.5 ,   8.76,   1.  ],
       [308.  , 106.  ,   3.  ,   3.  ,   3.  ,   8.24,   0.  ],
       [305.  , 103.  ,   2.  ,   2.5 ,   3.5 ,   8.13,   0.  ],
       [295.  ,  96.  ,   2.  ,   1.5 ,   2.  ,   7.34,   0.  ],
       [316.  ,  98.  ,   1.  ,   1.5 ,   2.  ,   7.43,   0.  ],
       [304.  ,  97.  ,   2.  ,   1.5 ,   2.  ,   7.64,   0.  ],
       [299.  ,  94.  ,   1.  ,   1.  ,   1.  ,   7.34,   0.  ],
       [302.  ,  99.  ,   1.  ,   2.  ,   2.  ,   7.25,   0.  ],
       [313.  , 101.  ,   3.  ,   2.5 ,   3.  ,   8.04,   0.  ],
       [318.  , 107.  ,   3.  ,   3.  ,   3.5 ,   8.27,   1.  ],
       [325.  , 110.  ,   4.  ,   3.5 ,   4.  ,   8.67,   1.  ],
       [303.  , 100.  ,   2.  ,   3.  ,   3.5 ,   8.06,   1.  ],
       [300.  , 102.  ,   3.  ,   3.5 ,   2.5 ,   8.17,   0.  ],
       [297.  ,  98.  ,   2.  ,   2.5 ,   3.  ,   7.67,   0.  ],
       [317.  , 106.  ,   2.  ,   2.  ,   3.5 ,   8.12,   0.  ],
       [327.  , 109.  ,   3.  ,   3.5 ,   4.  ,   8.77,   1.  ],
       [301.  , 104.  ,   2.  ,   3.5 ,   3.5 ,   7.89,   1.  ],
       [314.  , 105.  ,   2.  ,   2.5 ,   2.  ,   7.64,   0.  ],
       [321.  , 107.  ,   2.  ,   2.  ,   1.5 ,   8.44,   0.  ],
       [322.  , 110.  ,   3.  ,   4.  ,   5.  ,   8.64,   1.  ],
       [334.  , 116.  ,   4.  ,   4.  ,   3.5 ,   9.54,   1.  ],
       [338.  , 115.  ,   5.  ,   4.5 ,   5.  ,   9.23,   1.  ],
       [306.  , 103.  ,   2.  ,   2.5 ,   3.  ,   8.36,   0.  ],
       [313.  , 102.  ,   3.  ,   3.5 ,   4.  ,   8.9 ,   1.  ]]

    df = pd.DataFrame(data, columns = ['GRE Score', 'TOEFL Score','University Rating','SOP','LOR','CGPA','Research'])
    
    #df = pd.read_csv("https://github.com/ubaidkhan08/University-Acceptance-Predictor-on-Heroku/blob/main/Admission_Predict.csv")
    #df = df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
    #df = df.drop('Serial No.', axis=1)
    #XX = df.drop('Chance of Admit', axis=1)

    XX = df
    XX.iloc[0] = inputs
    XX = scalerrr.fit_transform(XX)

    predictionn = log_model.predict([XX[0]])
    predd = '{}'.format(predictionn)
    return(round(float(predd[1:7])*100))

def main():
    st.title("University Acceptance Predictor")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    </div>
    """

    gre=st.number_input('GRE Score')

    tofel=st.number_input('TOFEL Score')

    st.markdown(html_temp, unsafe_allow_html=True)
    sepal_length=st.slider('University Rating', 0.0, 5.0)
    sepal_width=st.slider('SOP Rating', 0.0, 5.0)
    petal_length=st.slider('LOR Rating', 0.0, 5.0)
    petal_width=st.slider('Select CGPA', 0.0, 10.0)
    #research=st.slider('Select Research', 0.0, 1.0)

    R = st.radio('Did Research?', ('Yes','No'))
    if R == 'Yes':
        research = 1
    else:
        research = 0

    inputs=np.array([[gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research]]).reshape(1, -1)
   

    if st.button('Predict My Chances'):
        output= classify(gre,tofel,sepal_length, sepal_width, petal_length, petal_width,research)
        st.success('Your chance of admission is: {}%'.format(output))


if __name__=='__main__':
    main()
