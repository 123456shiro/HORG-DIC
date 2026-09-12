# HORG-DIC

**Chirality-aware feature representation for digital image correlation initialization**

Companion C++ source code for the paper:

> **Chirality-Aware Initialization for Reliable Digital Image Correlation under Large Rotation and Heterogeneous Deformation**  
> Accepted for publication in *Measurement*.

This repository contains the HORG descriptor extraction and feature-matching source modules associated with the study. It is intended for researchers working on speckle-image correspondence and feature-based initialization for digital image correlation (DIC).

**Release scope:** this is a source-module release. It does not include a standalone DIC application, benchmark images, or the complete workflow needed to reproduce all results in the paper.

## Overview

Reliable initialization is essential for local DIC under large rotation and heterogeneous deformation. Ambiguous speckle correspondences can affect the initial affine estimate and the subsequent subpixel refinement.

The paper introduces the **histogram of oriented and rotated gradients (HORG)**, which augments a compact directional representation with a signed rotational cue:

- SIFT-style keypoint detection using Gaussian and Difference-of-Gaussian scale spaces.
- A **4 × 4 cell layout** around each keypoint.
- **Three directional branches separated by 120°** in each cell.
- **One additional signed rotational component per cell**.
- A **64-dimensional descriptor**: 48 directional components plus 16 rotational components.

Here, *chirality* refers to the signed cyclic ordering of local gradient directions in the image. It does not describe intrinsic material chirality or a physically chiral deformation field.

In the paper, HORG modifies the initial correspondence-construction stage. The retained correspondences support point-of-interest (POI) dependent affine initialization, followed by inverse compositional Gauss–Newton (IC-GN) refinement and full-field displacement reconstruction.

## Repository contents

| File | Description |
| --- | --- |
| [`my_sift.h`](my_sift.h) | Declaration of `opencorr::MySift` and feature-detection/descriptor parameters. |
| [`my_sift.cpp`](my_sift.cpp) | Scale-space construction, keypoint detection, orientation assignment, and descriptor computation. |
| [`my_match.h`](my_match.h) | Declaration of `opencorr::myMatch` and matching-related parameters. |
| [`my_match.cpp`](my_match.cpp) | Descriptor-search routines, ratio filtering, separate RANSAC and transformation-fitting routines, and visualization utilities. |
| [`README.md`](README.md) | Method overview, integration example, implementation notes, and data availability information. |

The source files use the `opencorr` namespace. The four supplied source/header files depend on OpenCV and the C++ standard library; they do not directly include OpenCorr or Eigen headers.

The following are outside this release:

- A `main()` entry point, Visual Studio solution, or CMake build configuration.
- POI-neighborhood selection and the complete local affine-initialization driver.
- IC-GN refinement, full-field displacement/strain reconstruction, and stereo reconstruction workflows.
- Benchmark images, reference displacement fields, comparison-method implementations, and experiment scripts.

## Requirements and integration

The original source-package documentation reports **Visual Studio 2019, C++, and OpenCV 4.5.0**. The paper additionally reports **Eigen 3.3.9** for the broader experimental implementation; Eigen is not directly required by the four files in this repository.

To integrate these modules into a Windows C++ project:

1. Configure a Visual Studio C++ project with OpenCV 4.5.0 and a compiler mode supporting C++11 or later.
2. Add `my_sift.cpp`, `my_sift.h`, `my_match.cpp`, and `my_match.h` to the project.
3. Configure the OpenCV include and library directories, link the appropriate OpenCV libraries for the selected build configuration and architecture, and make their runtime DLLs available to the executable.
4. Add your own application entry point to load images and call the modules. An example is provided below.

The source contains Windows-style OpenCV include paths. Porting to Linux or macOS requires adapting these paths and validating compiler/OpenCV compatibility. Comments and some string literals in the supplied source use a legacy Chinese encoding; preserve that encoding or convert the files consistently before changing the compiler's source-character-set setting.

These instructions describe integration with the supplied source. A fresh compilation and execution of the example have not been verified as part of this documentation update.

## Example: descriptors and initial correspondences

Save the following example as `main.cpp` in your configured project. It reads two **8-bit grayscale** images, extracts descriptors, performs OpenCV FLANN matching with two neighbors, and applies a **0.75 ratio threshold**. FLANN with `k = 2` and a ratio threshold of 0.75 are the matching choices reported in the paper.

The example uses OpenCV's matcher directly. It does not invoke the diagnostic image-writing code in `myMatch::match()` or estimate a DIC displacement field. The matcher interface is documented in the [OpenCV 4.5.0 reference](https://docs.opencv.org/4.5.0/db/d39/classcv_1_1DescriptorMatcher.html).

```cpp
#include "my_sift.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: horg_demo reference.png deformed.png\n";
        return 1;
    }

    try {
        cv::Mat reference = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        cv::Mat deformed = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        if (reference.empty() || deformed.empty()) {
            std::cerr << "Could not read the input images.\n";
            return 1;
        }

        opencorr::MySift extractor;
        std::vector<std::vector<cv::Mat>> gauss_ref, dog_ref;
        std::vector<std::vector<cv::Mat>> gauss_def, dog_def;
        std::vector<cv::KeyPoint> keys_ref, keys_def;
        cv::Mat desc_ref, desc_def;

        extractor.detect(reference, gauss_ref, dog_ref, keys_ref);
        extractor.comput_des(gauss_ref, keys_ref, desc_ref);
        extractor.detect(deformed, gauss_def, dog_def, keys_def);
        extractor.comput_des(gauss_def, keys_def, desc_def);

        if (desc_ref.empty() || desc_def.rows < 2) {
            std::cerr << "Insufficient descriptors for two-neighbor matching.\n";
            return 1;
        }
        if (!cv::checkRange(desc_ref) || !cv::checkRange(desc_def)) {
            std::cerr << "Non-finite descriptor values detected.\n";
            return 1;
        }

        cv::FlannBasedMatcher matcher;
        std::vector<std::vector<cv::DMatch>> candidates;
        matcher.knnMatch(desc_ref, desc_def, candidates, 2);

        std::vector<cv::DMatch> initial_matches;
        std::vector<cv::Point2f> reference_points, deformed_points;
        for (const auto& pair : candidates) {
            if (pair.size() == 2 && pair[1].distance > 0.0f &&
                pair[0].distance / pair[1].distance < 0.75f) {
                initial_matches.push_back(pair[0]);
                reference_points.push_back(keys_ref[pair[0].queryIdx].pt);
                deformed_points.push_back(keys_def[pair[0].trainIdx].pt);
            }
        }

        std::cout << "Reference descriptors: " << desc_ref.rows << '\n'
                  << "Deformed descriptors: " << desc_def.rows << '\n'
                  << "Descriptor dimension: " << desc_ref.cols << '\n'
                  << "Ratio-filtered pairs: " << initial_matches.size() << '\n';

        // reference_points[i] corresponds to deformed_points[i].
        // Select a local neighborhood for each POI before affine fitting.
        // Ratio-filtered matches have not yet been geometrically verified.
    }
    catch (const cv::Exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
```

If the resulting executable is named `horg_demo.exe`, run it with your own image paths:

```text
horg_demo.exe reference.png deformed.png
```

`reference.png` and `deformed.png` are illustrative filenames; these images are not bundled. The descriptor matrix has one row per keypoint, 64 columns, and type `CV_32FC1`. Correspondence counts depend on the images and parameters.

Use fresh pyramid and keypoint containers for each image. With the default `double_size = true`, `comput_des()` modifies keypoint coordinates to return them to the original image scale. Call it once on each freshly detected keypoint set before using the coordinates for matching or visualization.

## Main interfaces and defaults

| Interface | Purpose |
| --- | --- |
| `MySift::detect()` | Builds the Gaussian/DoG pyramids and detects keypoints. |
| `MySift::comput_des()` | Computes descriptors from the **Gaussian pyramid** and the detected keypoints. The spelling `comput_des` is the actual API name. |
| `myMatch::match_des()` | Custom descriptor-search routine; see the limitations below. |
| `myMatch::match()` | Ratio filtering and experiment-specific visualization; its internal RANSAC call is disabled. |
| `myMatch::ransac()` | Separate geometric-consistency estimation routine. |
| `myMatch::LMSget()` | Fits a transformation to paired coordinates. |

| Parameter | Default in the supplied headers |
| --- | --- |
| `nfeatures` | `0` (no explicit feature-count limit) |
| `nOctaveLayers` | `3` |
| `contrastThreshold` | `0.04` |
| `edgeThreshold` | `10` |
| `sigma` | `1.6` |
| `double_size` | `true` |
| `DESCR_WIDTH` | `4` |
| `DESCR_HIST_BINS` | `3` |
| `myMatch::dis_ratio` | `0.75` |

These are source defaults, not a complete record of the experimental settings used for every result in the paper.

## Implementation notes

### Descriptor computation

The active path is `comput_des()` → `calc_descriptors()` → `new_calc_sift_descriptor()`. Alternative descriptor routines are present, but their calls in `calc_descriptors()` are commented out.

Although `my_sift.h` defines a `DESCR_SCL_FCTR` constant of `3.0f`, `my_sift.cpp` subsequently defines a macro with the same name as `1.5f`. The subsequent descriptor code uses the macro value.

### Matching and geometric fitting

- `myMatch::match()` currently returns an empty transformation matrix because its RANSAC call is commented out. It does not populate `right_matchs` through an active inlier-extraction step.
- Its visualization code writes to hard-coded paths under `D://OpenCorr_VS2019/gauss/`, uses a fixed 512-pixel horizontal offset for some drawn lines, and selects 100 matches without checking that 100 are available. Adapt these operations before calling this wrapper on new images.
- The custom nearest-neighbor search in `match_des()` does not move the previous best candidate to second place when a new best candidate is found. The example above therefore uses OpenCV's FLANN matcher for the two-neighbor search.
- In the separate `ransac()` routine, `threshold` is compared with **squared** coordinate residuals. It should not be interpreted directly as a distance in pixels. The internal sampling loop also needs review for small or degenerate correspondence sets.
- For the implemented fitting branches, `LMSget()` estimates the mapping from the **second** coordinate set to the **first**. Check this direction when constructing reference-to-deformed DIC parameters. The string `"projective"` is accepted by validation but lacks a corresponding complete fitting/sampling branch; it should not be used as a substitute for `"perspective"`.

A single transformation fitted to all image correspondences does not implement the paper's POI-dependent initialization under heterogeneous deformation. The caller must select local correspondences and connect the local estimates to its DIC solver.

## Data and reproducibility

Benchmark images and third-party DIC datasets are **not included** in this repository. The paper evaluates the following cases:

| Dataset | Cases described in the paper |
| --- | --- |
| SEM 2D-DIC Challenge, Samples 1 and 3 | Rigid-body shift. |
| SEM 2D-DIC Challenge, Sample 9 | Rigid-body rotation and large-angle rotation. |
| SEM 2D-DIC Challenge, Sample 14 | Non-uniform deformation. |
| SEM 2D-DIC Challenge, Sample 12 | Tensile deformation of a plate with an open hole. |
| Stereo DIC Challenge 1.0, Sample 3 D-specimen | Three-dimensional large-deformation demonstration. |

Dataset access links for the completed 2D and stereo challenges are listed on the [official iDICs DIC Challenge page](https://www.idics.org/challenge). Consult the paper for the specific image selections and processing settings, and obtain the datasets from their original providers.

Reproducing the full-field results additionally requires the downstream DIC implementation, POI and region-of-interest definitions, local correspondence selection, refinement settings, and evaluation procedures. Running the descriptor example alone does not reproduce the paper's displacement-error tables.

## Data availability

The benchmark datasets used in this study are publicly available from the SEM 2D-DIC Challenge and the Stereo DIC Challenge. 

## Citation

If this code contributes to your research, please cite the associated paper:

> *Chirality-Aware Initialization for Reliable Digital Image Correlation under Large Rotation and Heterogeneous Deformation*. **Measurement**, accepted for publication.
