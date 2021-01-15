**Binary TTC: A Temporal Geofence for Autonomous Navigation**<br>
Abhishek Badki, Orazio Gallo, Jan Kautz, and Pradeep Sen<br>

## Abstract: 
*Time-to-contact (TTC), the time for an object to collide with the observer's plane, is a powerful tool for path planning: it is potentially more informative than the depth, velocity, and acceleration of objects in the scene---even for humans. TTC presents several advantages, including requiring only a monocular, uncalibrated camera. However, regressing TTC for each pixel is not straightforward, and most existing methods make over-simplifying assumptions about the scene. We address this challenge by estimating TTC via a series of simpler, binary classifications. We predict with* low latency *whether the observer will collide with an obstacle* within a certain time, *which is often more critical than knowing exact, per-pixel TTC. For such scenarios, our method offers a temporal geofence in 6.4 ms---over 25x faster than existing methods. Our approach can also estimate per-pixel TTC with arbitrarily fine quantization (including continuous values), when the computational budget allows for it. To the best of our knowledge, our method is the first to offer TTC information (binary or coarsely quantized) at sufficiently high frame-rates for practical use.*

## Paper:
http://arxiv.org/abs/2101.04777<br>

## Videos:<br>
<a href="https://youtu.be/uUQJcjyerM4">
  <img src="https://img.youtube.com/vi/uUQJcjyerM4/0.jpg" width="300"/>
</a>


    @InProceedings{badki2021BiTTC,
    author = {Badki, Abhishek and Gallo, Orazio and Kautz, Jan and Sen, Pradeep},
    title = {{B}inary {TTC}: {A} Temporal Geofence for Autonomous Navigation},
    booktitle = {arXiv preprint	arXiv:2101.04777},
    year = {2021}
    }
