import shtns
import pysht.geometry as geometry
import ducc0
ntheta = ducc0.fft.good_size(2500 + 2)
nphihalf = ducc0.fft.good_size(2500 + 1)
nphi = 2 * nphihalf
print(ntheta, nphi)

lmax = 1000
constructor = shtns.sht(int(lmax), int(lmax))
ntheta, nphi = 2520, 5040
constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.sht_reg_poles + shtns.SHT_THETA_CONTIGUOUS, nlat=ntheta, nphi=nphi)

print(constructor.print_info())