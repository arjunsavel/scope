FAQ
----

* Do I need a supercomputing cluster to run this code?
    Not necessarily. Computing Kp--Vsys maps at the correct resolution and over the correct range will likely take ~15 hours on a laptop, though.
* How much significance is "good enough"?
    Usually an S/N > ~4.5 is reasonable when it occurs at the expected velocity of the planet.
* Should I recalculate my simulated data with multiple different seeds?
    Yes, especially if you have relatively low S/N (lower than roughly 8). Random photon noise induces significant spread in the expected cross-correlation S/N.
* How do I choose the photon SNR?
    This comes from detailed exposure time calculator (ETC) runs. E.g., for IGRINS on GEMINI-S: https://igrins-jj.firebaseapp.com/etc/simple
