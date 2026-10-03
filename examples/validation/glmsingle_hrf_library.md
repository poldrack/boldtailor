# GLMsingle HRF library

`glmsingle_hrf_library.tsv` is the 20-HRF library used by GLMsingle and the
Natural Scenes Dataset, copied unchanged from
`glmsingle/hrf/getcanonicalhrflibrary.tsv` in the GLMsingle repository
(https://github.com/cvnlab/GLMsingle, commit 1ab54a6, 2025-11-09). The file is
501 rows by 20 columns: rows are a 0.1 s grid from 0 to 50 s after stimulus
onset, columns are the HRFs. The values are not peak-normalised (GLMsingle
rescales after convolving with the stimulus duration); the notebook divides
each column by its maximum before plotting.

The library is Copyright (c) 2021 Kendrick Kay and is redistributed here under
the BSD 3-Clause License of the GLMsingle repository. It is used in
`hrf_library_similarity.ipynb` for visual comparison with the sampled libraries
and is not part of the `boldtailor` package.
