# GroundSeg-Clustering-OptimizedKdtree

### 1.change the "root dir name" in "main.py" to the absolute path of your velodyne data, and run "main.py"
   `python main.py`

### 2.project including [ground segemantation]( /), [downsampling]( /), [clustering]( /), and a [optimized version of kdtree]( /).

### 3. The optimazation of kdtree mainly include:

  **a) [split tree along dimension with max variance]( /) instead of so called "round axis" or random axis**

  **b) [sorting part of data in each dimension]( /)**

<img src="https://user-images.githubusercontent.com/38379703/144193381-e9574be0-e31d-4908-86f7-74023f1ca459.png" width=400 height=512 title = "origin data"/> <img src="https://user-images.githubusercontent.com/38379703/144193411-1d2094d6-988d-4e0a-ae81-e67a9dfc6452.png" width=512 height=550 title = "ground points / ">
<img src="https://user-images.githubusercontent.com/38379703/144193430-15223a2e-5611-411f-ba9c-ef926a0ccad7.png" width=480 height=512 title = "non ground points" /> <img src="https://user-images.githubusercontent.com/38379703/144193439-c692ab4a-6a8f-4088-8d35-1a0f06ca54b1.png" width=480 height=512 title = "sampled non ground points" />
<img src="https://user-images.githubusercontent.com/38379703/144193454-9904c17e-f7bb-4858-8c2f-6847dcd0dbab.png" width=1024 height=512 title = "clusters" />


