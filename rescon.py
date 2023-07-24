import argparse
import gc
import numpy as np
import pandas as pd
import pathlib
import pickle
from collections import deque
from itertools import chain
from tqdm.auto import tqdm
import sys


def parse_msp(msp_file):
    """Parse an MSP file to a pandas.DataFrame
    Accepts a file object of an MSP file and parses the contents of the file
    into a pandas.DataFrame. This function is adapted from one originally by
    Adamo Young.

    :param msp_file:    a connection to the input .MSP file

    :return:    None
    """
    library = []
    spectrum = {}
    read_ms = False
    # tqdm generates a progress bar but doesn't show % progress, only number of
    # lines iterated so far; it doesn't know how many lines the file being
    # streamed in should have
    for line in tqdm(msp_file, desc="Parsing lines from the MSP file..."):
        line = line.strip("\n")
        # End of peak list - add this parsed spectrum to the batch list
        if len(line) == 0:
            # Break peak lines into (m/z, intensity, annotation) parts
            peak_list = [x.split() for x in spectrum["ms"]]
            if any([len(peak) > 2 for peak in peak_list]):
                peak_list = ["|".join([peak[0], peak[1], " ".join(peak[2:])])
                             for peak in peak_list]
            else:
                peak_list = ["|".join([peak[0], peak[1], '""'])
                             for peak in peak_list]
            spectrum["ms"] = peak_list
            library.append(spectrum.copy())
            spectrum = {}
            read_ms = False
        # Not end of peak list, but currently parsing peaks
        elif read_ms:
            if "; " in line:
                for peak in line.split(";"):
                    spectrum["ms"].append(peak)
            else:
                spectrum["ms"].append(line.replace(";", " "))
        # Other metadata lines in an entry
        else:
            # Next lines to parse will be peaks
            if line[0].isdigit():
                spectrum['ms'] = []
                if "; " in line:
                    for peak in line.split(";"):
                        spectrum["ms"].append(peak)
                else:
                    spectrum["ms"].append(line.replace(";", " "))
                read_ms = True
            else:
                line_split = line.split(': ')
                if line_split[0] in ['ID', 'CASNO', 'NISTNO', 'MW', "Link",
                                     'Num peaks']:
                    spectrum[line_split[0]] = int(line_split[1])
                elif line_split[0] in ['ExactMass']:
                    spectrum[line_split[0]] = float(line_split[1])
                elif line_split[0] in ['Spectrum_type']:
                    spectrum[line_split[0]] = int(line_split[1][-1])
                else:
                    try:
                        spectrum[line_split[0]] = line_split[1]
                    except:
                        print("Error parsing the following line:")
                        print(line_split)
                        print(line)
                        exit()
    msp_df = pd.DataFrame(library)
    # Specify column dtypes and process some fields
    msp_df["Name"] = msp_df["Name"].astype("string")
    msp_df["Notes"] = msp_df["Notes"].astype("string")
    msp_df["Precursor_type"] = msp_df["Precursor_type"].astype("string")
    msp_df["Spectrum_type"] = msp_df["Spectrum_type"].astype("category")
    msp_df["PrecursorMZ"] = msp_df["PrecursorMZ"].astype("string")
    msp_df["Instrument_type"] = msp_df["Instrument_type"].astype("category")
    msp_df["Instrument"] = msp_df["Instrument"].astype("category")
    msp_df["Sample_inlet"] = msp_df["Sample_inlet"].astype("category")
    msp_df["Ionization"] = msp_df["Ionization"].astype("category")
    msp_df["In-source_voltage"] = msp_df["In-source_voltage"].apply(
        lambda x: x.split("-")[-1] if not pd.isnull(x) else x
    ).astype("float")
    msp_df["Collision_gas"] = msp_df["Collision_gas"].astype("category")
    msp_df["Collision_energy"] = msp_df["Collision_energy"].astype("string")
    msp_df["Ion_mode"] = msp_df["Ion_mode"].astype("category")
    msp_df["InChIKey"] = msp_df["InChIKey"].astype("string")
    msp_df["Synon"] = msp_df["Synon"].astype("string")
    msp_df["Formula"] = msp_df["Formula"].astype("string")
    msp_df["MW"] = msp_df["MW"].astype("float")
    msp_df["ExactMass"] = msp_df["ExactMass"].astype("float")
    msp_df["CASNO"] = msp_df["CASNO"].astype("float")
    msp_df["NISTNO"] = msp_df["NISTNO"].astype("int")
    msp_df["ID"] = msp_df["ID"].astype("int")
    msp_df["Comment"] = msp_df["Comment"].astype("string")
    msp_df["Num peaks"] = msp_df["Num peaks"].astype("int")
    msp_df["ms"] = msp_df["ms"].astype("object")
    msp_df["Related_CAS#"] = msp_df["Related_CAS#"].astype("float")
    msp_df["msN_pathway"] = msp_df["msN_pathway"].astype("string")
    msp_df["Link"] = msp_df["Link"].astype("float")
    msp_df.set_index("ID")
    return msp_df


def generate_subsets(hr_df, lr_df, hr_Instrument_type, lr_Instrument_type, id_col):
    """
    Create subsets of precursor ions represented by HR-MS2 spectra and by
    HR-/LR-MSn spectral trees.

    :param hr_df:           a pd.DataFrame holding high-resolution-MS2 spectra
    :param lr_df:           a pd.DataFrame holding low-resolution-MSn spectra
    :param hr_Instrument_type:  Instrument_type field value for high-
                                -resolution-MS2 spectra
    :param lr_Instrument_type:  Instrument_type field value for low-resolution-
                                -MSn spectra
    :param id_col:          the name of the column holding unique spectrum ID
                            values (in NIST20, this is the NISTNO column)

    :return:    a dictionary holding pd.DataFrames describing MS2 spectra that
                lead to HR-/LR-MSn spectral trees and that are represented by
                HR-MS2 spectra
    """
    # Find precursor ions that are not associated with an InChIKey
    hr_no_inchikey = (
        hr_df.loc[
            [pd.isnull(inchikey) for inchikey in hr_df["InChIKey"].values],
            ["Name", "Precursor_type"]
        ]
        .drop_duplicates()
    )
    lr_no_inchikey = (
        lr_df.loc[
            [pd.isnull(inchikey) for inchikey in lr_df["InChIKey"].values],
            ["Name", "Precursor_type"]
        ]
        .drop_duplicates()
    )
    all_no_inchikey = (
        hr_no_inchikey.merge(
            lr_no_inchikey, how="outer", indicator=True
        )
        .rename(columns={"_merge": "Found_in"})
    )
    all_no_inchikey["Found_in"] = all_no_inchikey["Found_in"].map(
        {"left_only": "hr_df", "right_only": "lr_df", "both": "both"}
    )
    # Drop these precursor ions
    print(
        f"Dropped {hr_df['InChIKey'].apply(pd.isnull).sum()} HR spectra "
        "without InChI keys..."
    )
    hr_df = hr_df.dropna(subset=["InChIKey"])
    print(
        f"Dropped {lr_df['InChIKey'].apply(pd.isnull).sum()} LR spectra "
        "without InChI keys..."
    )
    lr_df = lr_df.dropna(subset=["InChIKey"])
    # Get (InChIKey, Precursor_type) tuples that are represented by HR-MS2
    hr_ms2_ions = (
        hr_df.query(
            f"Instrument_type == '{hr_Instrument_type}' and Spectrum_type == 2"
        )
        .loc[:, ["InChIKey", "Name", "Precursor_type"]]
    )
    hr_ms2_ions = hr_ms2_ions.drop_duplicates(
        subset=["InChIKey", "Precursor_type"]
    )
    # Get IDs of MS2 spectra that lead to LR-MSn spectral trees
    lr_msn_ids = (
        lr_df
        .query(f"Instrument_type == '{lr_Instrument_type}' and Spectrum_type == 3")
        ["Link"]
        .drop_duplicates()
        .values
    )
    lr_msn_ions = lr_df.loc[
        [sp in lr_msn_ids for sp in lr_df[arg.id_col].values],
        ["InChIKey", "Name", "Precursor_type", id_col]
    ]
    # Return these DataFrames of precursor ions in a dictionary
    subsets = {"hr_ms2_ions": hr_ms2_ions, "lr_msn_ions": lr_msn_ions}
    return subsets


def extract_msn_spectral_tree(ms2_sp_id, id_col, sp_df=None):
    """Extract an MSn spectral tree given an MS2 spectrum's ID"""
    # Get MS2 spectrum
    ms2_sp = sp_df.query(f"{id_col} == {ms2_sp_id}")
    assert ms2_sp.shape[0] == 1, (
        f"Spectrum ID must denote a single spectrum, but {ms2_sp.shape} were "
        "returned"
    )
    assert ms2_sp["Spectrum_type"].values[0] == 2, (
        "Spectrum ID must denote an MS2 spectrum"
    )
    # Get MS3 and MS4 spectra
    ms3_sp = sp_df.loc[sp_df["Link"].isin(ms2_sp["NISTNO"])]
    ms4_sp = sp_df.loc[sp_df["Link"].isin(ms3_sp["NISTNO"])]
    return pd.concat([ms2_sp, ms3_sp, ms4_sp])


def convert_peak_resolution(
            sp_tree, hr_ms2_sp_df, lr_msn_sp_df, hr_Instrument_type,
            lr_Instrument_type, tolerance, max_donors, id_col
        ):
    """Convert peak resolutions in native LA-MSn spectra

    :param precursor_ion:   a pandas.Series() holding an (InChIKey,
                            Precursor_type) pair
    :param subset_df:       a pandas.DataFrame() of fragmentation spectra
    :param tolerance:       a float denoting the mass tolerance for conversion

    :return:                a pandas.DataFrame() of converted fragmentation
                            spectra
    """
    ion_InChIKey = sp_tree["InChIKey"]
    ion_Precursor_type = sp_tree["Precursor_type"]
    # Get the MSn spectra
    ion_lr_msn_sp_df = extract_msn_spectral_tree(
        sp_tree[id_col], id_col=id_col,
        sp_df=lr_msn_sp_df
    )
    # Get a list of donor m/z values
    ion_hr_ms2_sp_df = hr_ms2_sp_df.query(
        f"InChIKey == '{ion_InChIKey}' and "
        f"Precursor_type == '{ion_Precursor_type}' and "
        f"Instrument_type == '{hr_Instrument_type}' and "
        "Spectrum_type == 2"
    )
    donor_composite_mz = np.array(
        list(
            chain.from_iterable(
                [
                    [float(peak.split("|")[0]) for peak in donor_peak_list]
                    for donor_peak_list in ion_hr_ms2_sp_df["ms"]
                ]
            )
        )
    )
    donor_composite_intensity = np.array(
        list(
            chain.from_iterable(
                [
                    [float(peak.split("|")[1]) for peak in donor_peak_list]
                    for donor_peak_list in ion_hr_ms2_sp_df["ms"]
                ]
            )
        )
    )
    # Get a list of recipient peak lists
    recipient_peak_lists = ion_lr_msn_sp_df["ms"]
    # For every peak throughout the MSn spectra in this spectral tree..
    converted_peak_lists = []
    for peak_list in recipient_peak_lists:
        converted_peak_list = []
        for peak in peak_list:
            peak_mz = float(peak.split("|")[0])
            # Find the closest donor peak m/z and if it's within tolerance,
            # replace the low-resolution m/z value
            peak_differences = np.abs(donor_composite_mz - peak_mz)
            n_donors = (peak_differences < tolerance).sum()
            too_many_donors = max_donors is not None and n_donors < max_donors
            if peak_differences.min() < tolerance and not too_many_donors:
                candidate_donors = list(
                    zip(
                        donor_composite_mz[peak_differences < tolerance],
                        donor_composite_intensity[peak_differences < tolerance]
                    )
                )
                # [0] is m/z; [1] is intensity
                most_intense_peak_mz = candidate_donors[
                    np.argmax([peak[1] for peak in candidate_donors])
                ][0]
                converted_peak_list.append(
                    "|".join(
                        [
                            str(most_intense_peak_mz),
                            "|".join(peak.split("|")[1:]),
                            "True",
                            str(n_donors)
                        ]
                    )
                )
            else:
                converted_peak_list.append(
                    "|".join(
                        [
                            str(peak_mz),
                            "|".join(peak.split("|")[1:]),
                            "False",
                            str("|")
                        ]
                    )
                )
        converted_peak_lists.append(converted_peak_list)
    # Store the augmented peak lists as a new column and return the spectra as
    # a pd.DataFrame
    ion_lr_msn_sp_df["Converted_ms"] = converted_peak_lists
    return(ion_lr_msn_sp_df)


def get_converted_peaks(sp_tree, lr_Instrument_type, id_col, con_sp_df):
    """Get DataFrames of peaks that were converted in MSn spectra

    :param precursor ion:   a pandas.Series() that holds a Converted_ms field
    :param subset_df:       a pandas.DataFrame() of converted fragmentation
                            spectra

    :return:                a dictionary holding pandas.DataFrames of peak
                            lists showing matched peaks between spectra of this
                            particular precursor ion
    """
    # Subset for the converted MSn spectra belonging to this precursor ion
    ion_InChIKey = sp_tree["InChIKey"]
    ion_con_sp_df = extract_msn_spectral_tree(
        sp_tree[id_col], id_col, sp_df=con_sp_df
    )
    assert ion_con_sp_df.query("Spectrum_type == 2").shape[0] == 1, (
        "More than one MS2 extracted for an MSn spectral tree!"
    )
    # Create a list holding rows of data for peaks found in this spectral tree
    msn_peaks_df = []
    for i, row in ion_con_sp_df.iterrows():
        converted_peaks = [peak.split("|") for peak in row["Converted_ms"]]
        original_peaks = [peak.split("|") for peak in row["ms"]]
        assert len(converted_peaks) == len(original_peaks)
        for i in range(0, len(converted_peaks)):
            converted_mz = (
                float(converted_peaks[i][0]) \
                if eval(converted_peaks[i][3]) else np.nan
            )
            n_donor_candidates = (
                int(converted_peaks[i][4]) \
                if converted_peaks[i][4] != "" else np.nan
            )
            msn_peaks_df.append(
                [
                    row["Name"], ion_InChIKey, row["Precursor_type"],
                    row["Spectrum_type"], row[id_col], sp_tree[id_col],
                    float(original_peaks[i][0]), converted_mz,
                    float(converted_peaks[i][1]), converted_peaks[i][2],
                    eval(converted_peaks[i][3]), n_donor_candidates
                ]
            )
    # If this passes for every converted spectrum, every peak is accounted for
    n_peaks = ion_con_sp_df["Num peaks"].sum()
    assert len(msn_peaks_df) == ion_con_sp_df["Num peaks"].sum(), (
        "Number of peaks reported in MSP does not match count after conversion!"
    )
    # Form a DataFrame from the list of rows formed above
    peaks_df = pd.DataFrame(
        msn_peaks_df, columns=[
            "Name", "InChIKey", "Precursor_type", "Spectrum_type", id_col,
            f"MS2_{id_col}", "lr_m/z", "Converted_m/z", "Intensity",
            "Annotation", "Converted", "Num_Donor_Candidates"
        ]
    )
    # MS2 spectra will have nan for Link values -- transplant their IDs instead
    peaks_df[pd.isnull(peaks_df[f"MS2_{id_col}"])].fillna(
        peaks_df[id_col], inplace=True
    )
    return peaks_df


def write_msp(sp, msp_file):
    """Write a pd.Series of a spectrum to an MSP file
    Accepts a file object to an MSP file and writes the contents of a pd.Series
    to the file.

    :param sp:       a pd.Series holding a spectrum
    :param msp_file:    a connection to the output .MSP file

    :return:    None
    """
    fields = [
        "Name", "Notes", "Precursor_type", "Spectrum_type", "PrecursorMZ",
        "Instrument_type", "Instrument", "Sample_inlet", "Ionization",
        "In-source_voltage", "Collision_gas", "Collision_energy", "Ion_mode",
        "InChIKey", "Synon", "Formula", "MW", "ExactMass", "CASNO", "NISTNO",
        "ID", "Comment", "Special_fragmentation", "Related_CAS#", "Link",
        "msN_pathway", "Num peaks", "Converted_ms"
    ]
    data = sp.to_dict()
    for f in fields:
        if f not in data.keys():
            continue
        else:
            if f == "Converted_ms":
                for peak in data[f]:
                    peak = peak.split("|")
                    converted = "; Converted" if eval(peak[3]) else ""
                    msp_file.write(
                        " ".join(peak[0:2]) + " " + peak[2][:-1] + converted + '"\n'
                    )
                msp_file.write("\n")
            elif f == "PrecursorMZ":
                msp_file.write(
                    f+ ": " + ", ".join([str(mz) for mz in data[f]]) + "\n"
                )
            elif not pd.isnull(data[f]):
                msp_file.write(f + ": " + str(data[f]) + "\n")


if __name__ == "__main__":
    # Parse and check command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hr_msp", type=str, required=True,
        help="the path to the MSP file holding high-resolution-MS2 spectra"
    )
    parser.add_argument(
        "--lr_msp", type=str, required=True,
        help="the path to the MSP file holding low-resolution-MSn spectra"
    )
    parser.add_argument(
        "--out_msp", type=str, required=True,
        help="the path to the output MSP file holding converted-MSn spectra"
    )
    parser.add_argument(
        "--hr_Instrument_type", type=str, required=True,
        help="Instrument_type field value for high-resolution-MS2 spectra"
    )
    parser.add_argument(
        "--lr_Instrument_type", type=str, required=True,
        help="Instrument_type field value for low-resolution-MSn spectra"
    )
    parser.add_argument(
        "--id_col", type=str, required=True,
        help="name of the column holding unique spectrum ID values "
        "(in NIST20, this is the NISTNO column)"
    )
    parser.add_argument(
        "--tol", type=float, default=0.1,
        help="+/- tolerance to use when searching for donor peaks (units: Th)"
    )
    parser.add_argument(
        "--max_donors", type=int, default=None,
        help="max number of HR peaks allowed in window during conversion"
    )
    parser.add_argument(
        "--hr_df", type=str, default=None,
        help=(
            "export a serialized pandas.DataFrame holding the parsed "
            "high-resolution-MS2 spectra to the specified file path"
        )
    )
    parser.add_argument(
        "--lr_df", type=str, default=None,
        help=(
            "export a serialized pandas.DataFrame holding the parsed "
            "low-resolution-MSn spectra to the specified file path"
        )
    )
    parser.add_argument(
        "--con_df", type=str, default=None,
        help=(
            "export a serialized pandas.DataFrame holding the converted-MSn"
            "spectra to the specified file path"
        )
    )
    parser.add_argument(
        "--peaks_df", type=str, default=None,
        help=(
            "export a serialized pandas.DataFrame describing "
            "low-resolution/converted peaks to the specified file path"
        )
    )
    arg = parser.parse_args()

    # Check that files/results dir exist
    HR_FP = pathlib.Path(arg.hr_msp)
    LR_FP = pathlib.Path(arg.lr_msp)
    assert \
        HR_FP.is_file(), "MSP file with high-resolution-MS2 spectra not found"
    assert \
        LR_FP.is_file(), "MSP file with low-resolution-MSn spectra not found"

    # Load databases into pd.DataFrames
    ### FIXME: When these are called, the conversion on the LA-RES-MSn spectra
    ###        goes from ~30 mins to ~2 hours --> probably a memory issue
#    with open(HR_FP, "r") as f:
#        hr_df = parse_msp(f)
#    with open(LR_FP, "r") as f:
#        lr_df = parse_msp(f)
    # For now, run the parsing code separately, serialize the results, and
    # import here
    with open("./hr_df.pickle", "rb") as f:
        hr_df = pickle.load(f)
    with open("./lr_df.pickle", "rb") as f:
        lr_df = pickle.load(f)

    # Subset the databases for just the relevant spectra
    hr_df = hr_df.query(f"Instrument_type == '{arg.hr_Instrument_type}'")
    lr_df = lr_df.query(f"Instrument_type == '{arg.lr_Instrument_type}'")

    # Save the parsed databases as serialized pandas.DataFrames if requested
    if arg.hr_df is not None:
        hr_df.to_pickle(pathlib.Path(arg.hr_df))
    if arg.lr_df is not None:
        lr_df.to_pickle(pathlib.Path(arg.lr_df))

    assert arg.id_col in hr_df.columns and arg.id_col in lr_df.columns, (
        f"Could find column {arg.id_col} after parsing spectra!"
    )

    # Get subsets of ions
    subsets = generate_subsets(
        hr_df, lr_df, arg.hr_Instrument_type, arg.lr_Instrument_type,
        arg.id_col
    )

    # Determine subset of overlapping ions between HR-MS2 and LR-MSn
    common_ions_df = pd.merge(
        subsets["hr_ms2_ions"], subsets["lr_msn_ions"],
        how="inner", on=["InChIKey", "Precursor_type"]
    ).drop(
        columns=["Name_y"]
    ).rename(
        columns={"Name_x": "Name"}
    )

    # Perform the peak conversion method
    tqdm.pandas(desc="Converting LR peaks...")
    converted_spectra = common_ions_df.progress_apply(
        convert_peak_resolution, hr_ms2_sp_df=hr_df, lr_msn_sp_df=lr_df,
        hr_Instrument_type=arg.hr_Instrument_type,
        lr_Instrument_type=arg.lr_Instrument_type, tolerance=arg.tol,
        max_donors=arg.max_donors, id_col=arg.id_col, axis=1
    )
    converted_spectra_df = pd.concat(
        converted_spectra.tolist(), ignore_index=True
    )
    # Save the converted database as a serialized pd.DataFrame() of spectra
    print("Saving converted spectra to pickle file...")
    if arg.con_df is not None:
        converted_spectra_df.to_pickle(
            pathlib.Path(arg.con_df)
        )
        
    # Write converted spectra to MSP file specified by arg.out_msp
    if arg.out_msp is not None:
        tqdm.pandas(desc="Writing converted spectra to MSP...")
        with open(arg.out_msp, "w") as f:
            converted_spectra_df.progress_apply(write_msp, msp_file=f, axis=1)
        
    # Don't bother computing the peaks table if not requested
    if arg.peaks_df is not None:
        # Add the DataFrames of converted spectra as a new column to the table
        # of common precursor ions
        common_ions_df["Converted_df"] = converted_spectra
        # Find peaks throughout each spectral tree that got converted
        tqdm.pandas(desc="Finding converted peaks in native LR-MSn spectra...")
        converted_peaks_df = pd.concat(
            common_ions_df.progress_apply(
                get_converted_peaks, lr_Instrument_type=arg.lr_Instrument_type, id_col=arg.id_col,
                con_sp_df=converted_spectra_df, axis=1
            ).values
        )

        # Save this as a pickle file
        converted_peaks_df.to_pickle(pathlib.Path(arg.peaks_df))
