# Decomposition-Based Optimal Bounds for Privacy Amplification via Shuffling

This repository contains the code for the paper:  
**"Decomposition-Based Optimal Bounds for Privacy Amplification via Shuffling."**

We develop an efficient algorithm to compute the optimal privacy amplification bounds in the shuffle model using decomposition-based methods. Our implementation supports a wide range of specific local randomizers and provides both upper and lower bounds with high precision.

### Dependencies
This project requires standard Python libraries, including:
- `NumPy`
- `SciPy`
- `matplotlib`

We also use `CuPy` to accelerate the Fast Fourier Transform (FFT) computations.

### How to Use
To reproduce the numerical experiments in the paper, simply run:
python test.py

### Baseline Implementations
This project incorporates and builds upon the following open-source codebases:

- [Balle et al., CRYPTO 2019](https://github.com/BorjaBalle/amplification-by-shuffling)
- [Feldman et al., FOCS 2021](https://github.com/apple/ml-shuffling-amplification)

### Note
This implementation is shared for reproducibility and review during the submission process.  
A more detailed README and documentation will be added upon publication.

---
For any questions or suggestions, feel free to open an issue or contact us.

