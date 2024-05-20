# Known sharp edges and nice-to-have's
 
 * update GPU code with synthesis() call - once it's available
 * 
 * check mmax parameter - not sure it actually does anything, or the right thing.
 * check adjointness of both Lenspyx and cunusht - debug_lenmap2gclm() looks good, but the tutorials give different adjoints, why is that?
 * dtypeing - not everything needs to be double precision, so could get some small speed ups here.
 * check speed adjoint_synthesis_general - it's somehwat slow, only 3 times speed up, should be 6, perhaps nuFFT dtype wrong.
 * shapeing - either make outward facing functions shape agnostic, or align shaping with DUCC convention.
 * align transformer init for both GPU and CPU - perhaps add a setup_lensing() for GPU?
 * tests and integration to CI - they use old interface, so surely not working anymore.
 * one pyproject.toml that installs everything - don't want separate cunusht/c/.. installation no longer