# v10 Generative Shape-Constraint Report

Version 10 separates geometry, appearance, and generation. The original 16-bit image supplies per-lamella center paths, subpixel endpoints, transverse FWHM, and adjacent interlayer dimensions. A structural condition image is then supplied to the built-in image generator together with the source and an unregistered appearance reference.

Soft image conditioning alone recovered 49/49 left and 50/51 right detected layers, but its endpoint P95 error remained 4.45 px. The workflow therefore adds an analytic hard projection: generative content supplies low-frequency background, brightness, and texture, while measured Gaussian ribbons supply transverse FWHM and logistic endpoint profiles supply length.

The source contains 49 left and 51 right lamellae and 98 interlayers. Median lamella width is 4.245 px, median length is 525.504 px, median interlayer width is 6.838 px, and median common interlayer length is 486.082 px.

Against the actual low-frequency condition coordinates, final endpoint-shift P95 is 0.083 px, length-change P95 is 0.111 px, median width error is 4.54%, and width-error P95 is 12.33%. The analytic projection contains all 100 layers; the ordinary peak detector merges the final close right-side pair, whose centers are only about 6.44 px apart.

The output remains generative and must not be treated as metrology truth. Use it for visual review and prototyping; use the non-generative measurement path with calibration for thickness, length, or defect acceptance.
