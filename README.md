# rescon

This repository holds code for `rescon`, a tool to convert *m/z* values of low-resolution MS peaks to a higher resolution using complementary fragmentation data. To use `rescon`, two sets of spectra are required: one comprised of low-resolution spectra to be converted and one of high-resolution spectra to source donor *m/z* values from. These spectra must be stored in the .MSP format.

## Usage

```
rescon.py [-h] --hr_msp HR_MSP --lr_msp LR_MSP --out_msp OUT_MSP --hr_Instrument_type HR_INSTRUMENT_TYPE --lr_Instrument_type LR_INSTRUMENT_TYPE --id_col ID_COL
                 [--tol TOL] [--max_donors MAX_DONORS] [--hr_df HR_DF] [--lr_df LR_DF] [--con_df CON_DF] [--peaks_df PEAKS_DF]
```

* `--hr_msp HR_MSP`: path to the .MSP file holding high-resolution spectra (str)
* `--lr_msp LR_MSP`: path to the .MSP file holding low-resolution spectra (str)
* `--out_msp OUT_MSP`: desired path to the .MSP file holding converted spectra (str)
* `--hr_Instrument_type HR_INSTRUMENT_TYPE`: value found in the `Instrument_type` field denoting high-resolution spectra (str)
* `--lr_Instrument_type LR_INSTRUMENT_TYPE`: value found in the `Instrument_type` field denoting low-resolution spectra (str)
* `--id_col ID_COL`: name of the field denoting spectral IDs (str)
* `[--tol TOL]`: value denoting +/- m/z tolerance to allow when converting LR peaks (float; default=0.1)
* `[--max_donors MAX_DONORS]`: optional value denoting max number of donors to allow when converting LR peaks (int; default=None)
* `[--hr_df HR_DF] [--lr_df LR_DF]`: if set, saves a seralized DataFrame holding high-/low-resolution spectra at the specified path (str; default=None)
* `[--con_df CON_DF]`: if set, saves a seralized DataFrame holding converted spectra at the specified path (str; default=None)
* `[--peaks_df PEAKS_DF]`: if set, saves a seralized DataFrame holding converted peaks at the specified path (str; default=None)
